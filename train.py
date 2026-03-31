"""
Copyright (C) 2018 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
from utils import get_all_data_loaders, prepare_sub_folder, write_html, write_loss, get_config, write_2images, Timer
import argparse
from torch.autograd import Variable
from trainer import MUNIT_Trainer, UNIT_Trainer
import torch.backends.cudnn as cudnn
import torch
try:
    from itertools import izip as zip
except ImportError:
    pass
import os
import sys
import csv
import numpy as np
import tensorboardX
import shutil
from multiprocessing import freeze_support

# ─── Import evaluasi FID & LPIPS ───────────────────────────
try:
    from pytorch_fid import fid_score
    FID_AVAILABLE = True
except ImportError:
    print("[PERINGATAN] pytorch-fid tidak ditemukan. FID dinonaktifkan.")
    FID_AVAILABLE = False

try:
    import lpips as lpips_lib
    LPIPS_AVAILABLE = True
except ImportError:
    print("[PERINGATAN] lpips tidak ditemukan. LPIPS dinonaktifkan.")
    LPIPS_AVAILABLE = False

from torchvision.utils import save_image


# ───────────────────────────────────────────────────────────
# FUNGSI EVALUASI
# ───────────────────────────────────────────────────────────

def setup_eval_log(log_path):
    """Buat file CSV log evaluasi jika belum ada."""
    expected_header = ["iterasi", "FID_AtoB", "LPIPS_AtoB"]
    if os.path.exists(log_path):
        with open(log_path, "r", newline="") as f:
            reader = csv.reader(f)
            current_header = next(reader, None)
        if current_header == expected_header:
            return
    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(expected_header)
    print(f"[EVAL] Log file dibuat: {log_path}")


def generate_eval_images(trainer, images_a, images_b, fake_dir_ab, real_dir_b, n_saved):
    """
    Generate gambar hasil translasi A→B dan B→A,
    simpan ke folder sementara untuk evaluasi FID & LPIPS.
    """
    os.makedirs(fake_dir_ab, exist_ok=True)
    os.makedirs(real_dir_b,  exist_ok=True)

    trainer.gen_a.eval()
    trainer.gen_b.eval()

    with torch.no_grad():
        outputs = trainer.sample(images_a.cuda(), images_b.cuda())
        # outputs[0] = a_real, outputs[1] = a_fake (B→A), outputs[2] = b_real, outputs[3] = b_fake (A→B)
        b_real, b_fake = outputs[2], outputs[3]

        for i in range(b_fake.size(0)):
            idx = n_saved + i
            save_image(b_fake[i], os.path.join(fake_dir_ab, f"{idx:05d}.jpg"), normalize=True)
            save_image(b_real[i], os.path.join(real_dir_b,  f"{idx:05d}.jpg"), normalize=True)

    trainer.gen_a.train()
    trainer.gen_b.train()

    return n_saved + b_fake.size(0)


def hitung_fid(real_dir, fake_dir, device):
    """Hitung FID score antara folder real dan fake."""
    if not FID_AVAILABLE:
        return -1.0
    try:
        score = fid_score.calculate_fid_given_paths(
            [real_dir, fake_dir],
            batch_size=50,
            device=device,
            dims=2048
        )
        return round(float(score), 4)
    except Exception as e:
        print(f"  [EVAL] FID error: {e}")
        return -1.0


def hitung_lpips(real_dir, fake_dir, loss_fn, device):
    """Hitung rata-rata LPIPS antara gambar real dan fake."""
    if not LPIPS_AVAILABLE:
        return -1.0
    try:
        from torchvision import transforms
        from PIL import Image

        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

        real_files = sorted(os.listdir(real_dir))
        fake_files = sorted(os.listdir(fake_dir))
        n          = min(len(real_files), len(fake_files), 100)  # max 100 pasang

        scores = []
        for i in range(n):
            img_real = transform(Image.open(os.path.join(real_dir, real_files[i])).convert("RGB")).unsqueeze(0).to(device)
            img_fake = transform(Image.open(os.path.join(fake_dir, fake_files[i])).convert("RGB")).unsqueeze(0).to(device)
            d        = loss_fn(img_real, img_fake)
            scores.append(d.item())

        return round(float(np.mean(scores)), 4)
    except Exception as e:
        print(f"  [EVAL] LPIPS error: {e}")
        return -1.0


def simpan_log(log_path, iterasi, fid_ab, lpips_ab):
    """Tulis satu baris hasil evaluasi ke CSV."""
    with open(log_path, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([iterasi, fid_ab, lpips_ab])
    print(f"  [EVAL] Iter {iterasi:08d} | FID A->B: {fid_ab} | LPIPS A->B: {lpips_ab}")


def bersihkan_folder_eval(*dirs):
    """Hapus isi folder eval sementara agar tidak menumpuk."""
    for d in dirs:
        if os.path.exists(d):
            shutil.rmtree(d)
            os.makedirs(d)


# ───────────────────────────────────────────────────────────
# MAIN
# ───────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',      type=str, default='configs/sasirangan.yaml')
    parser.add_argument('--output_path', type=str, default='.')
    parser.add_argument("--resume",      action="store_true")
    parser.add_argument('--trainer',     type=str, default='MUNIT', help="MUNIT|UNIT")
    opts = parser.parse_args()

    cudnn.benchmark = True

    config                  = get_config(opts.config)
    max_iter                = config['max_iter']
    display_size            = config['display_size']
    config['vgg_model_path'] = opts.output_path

    # ── Interval evaluasi (tambahkan di config yaml jika mau, default 1000) ──
    eval_interval = config.get('eval_iter', 1000)

    # ── Setup trainer ──
    if opts.trainer == 'MUNIT':
        trainer = MUNIT_Trainer(config)
    elif opts.trainer == 'UNIT':
        trainer = UNIT_Trainer(config)
    else:
        sys.exit("Only support MUNIT|UNIT")
    trainer.cuda()

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # ── Inisialisasi LPIPS ──
    loss_fn_lpips = lpips_lib.LPIPS(net="alex").to(DEVICE) if LPIPS_AVAILABLE else None

    # ── Data loaders ──
    train_loader_a, train_loader_b, test_loader_a, test_loader_b = get_all_data_loaders(config)
    train_display_images_a = torch.stack([train_loader_a.dataset[i] for i in range(display_size)])
    train_display_images_b = torch.stack([train_loader_b.dataset[i] for i in range(display_size)])
    test_display_images_a  = torch.stack([test_loader_a.dataset[i]  for i in range(display_size)])
    test_display_images_b  = torch.stack([test_loader_b.dataset[i]  for i in range(display_size)])

    # ── Setup folder output ──
    model_name          = os.path.splitext(os.path.basename(opts.config))[0]
    train_writer        = tensorboardX.SummaryWriter(os.path.join(opts.output_path + "/logs", model_name))
    output_directory    = os.path.join(opts.output_path + "/outputs", model_name)
    checkpoint_directory, image_directory = prepare_sub_folder(output_directory)
    shutil.copy(opts.config, os.path.join(output_directory, 'config.yaml'))

    # ── Setup folder & log evaluasi ──
    eval_dir    = os.path.join(output_directory, "eval_temp")
    fake_dir_ab = os.path.join(eval_dir, "fake_AtoB")
    real_dir_b  = os.path.join(eval_dir, "real_B")
    log_path    = os.path.join(output_directory, "eval_log.csv")
    setup_eval_log(log_path)

    # ── Start training ──
    iterations = trainer.resume(checkpoint_directory, hyperparameters=config) if opts.resume else 0

    while True:
        for it, (images_a, images_b) in enumerate(zip(train_loader_a, train_loader_b)):
            trainer.update_learning_rate()
            images_a, images_b = images_a.cuda().detach(), images_b.cuda().detach()

            with Timer("Elapsed time in update: %f"):
                trainer.dis_update(images_a, images_b, config)
                trainer.gen_update(images_a, images_b, config)
                torch.cuda.synchronize()

            # ── Log loss ──
            if (iterations + 1) % config['log_iter'] == 0:
                print("Iteration: %08d/%08d" % (iterations + 1, max_iter))
                write_loss(iterations, trainer, train_writer)

            # ── Simpan gambar ──
            if (iterations + 1) % config['image_save_iter'] == 0:
                with torch.no_grad():
                    test_image_outputs  = trainer.sample(test_display_images_a.cuda(),  test_display_images_b.cuda())
                    train_image_outputs = trainer.sample(train_display_images_a.cuda(), train_display_images_b.cuda())
                write_2images(test_image_outputs,  display_size, image_directory, 'test_%08d'  % (iterations + 1))
                write_2images(train_image_outputs, display_size, image_directory, 'train_%08d' % (iterations + 1))
                write_html(output_directory + "/index.html", iterations + 1, config['image_save_iter'], 'images')
                torch.cuda.empty_cache()

            if (iterations + 1) % config['image_display_iter'] == 0:
                with torch.no_grad():
                    image_outputs = trainer.sample(train_display_images_a.cuda(), train_display_images_b.cuda())
                write_2images(image_outputs, display_size, image_directory, 'train_current')
                torch.cuda.empty_cache()

            # ── Simpan checkpoint ──
            if (iterations + 1) % config['snapshot_save_iter'] == 0:
                trainer.save(checkpoint_directory, iterations)

            # ─────────────────────────────────────────────────
            # EVALUASI FID & LPIPS
            # ─────────────────────────────────────────────────
            if (iterations + 1) % eval_interval == 0:
                print(f"\n[EVAL] Mulai evaluasi pada iterasi {iterations + 1}...")

                # 1. Bersihkan folder eval sementara
                bersihkan_folder_eval(fake_dir_ab, real_dir_b)

                # 2. Generate gambar dari test loader
                n_saved = 0
                for eval_a, eval_b in zip(test_loader_a, test_loader_b):
                    n_saved = generate_eval_images(
                        trainer,
                        eval_a, eval_b,
                        fake_dir_ab, real_dir_b,
                        n_saved
                    )
                    if n_saved >= 200:  # cukup 200 gambar untuk evaluasi
                        break

                # 3. Hitung FID
                fid_ab = hitung_fid(real_dir_b, fake_dir_ab, DEVICE)

                # 4. Hitung LPIPS
                lpips_ab = hitung_lpips(real_dir_b, fake_dir_ab, loss_fn_lpips, DEVICE)

                # 5. Simpan ke CSV
                simpan_log(log_path, iterations + 1, fid_ab, lpips_ab)

                # 6. Catat ke TensorBoard
                train_writer.add_scalar("eval/FID_AtoB",   fid_ab,   iterations + 1)
                train_writer.add_scalar("eval/LPIPS_AtoB", lpips_ab, iterations + 1)
                torch.cuda.empty_cache()
                print(f"[EVAL] Selesai.\n")

            iterations += 1
            if iterations >= max_iter:
                sys.exit('Finish training')


if __name__ == '__main__':
    freeze_support()
    main()




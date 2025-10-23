from flask import Flask, render_template, request, redirect, url_for, send_from_directory, flash
import os
import time
import subprocess
from pathlib import Path
import sys
from werkzeug.utils import secure_filename
import shutil
import cv2
import numpy as np

app = Flask(__name__)
app.secret_key = 'vitonhd-web'

# Base paths (app is expected under VITON-HD/web/app.py)
APP_DIR = Path(__file__).resolve().parent
VITON_DIR = APP_DIR.parent
DATASETS_DIR = VITON_DIR / 'datasets'
CHECKPOINTS_DIR = VITON_DIR / 'checkpoints'
RESULTS_DIR = VITON_DIR / 'results'

TEST_DIR = DATASETS_DIR / 'test'
IMG_DIR = TEST_DIR / 'image'
CLOTH_DIR = TEST_DIR / 'cloth'

# Optional external preprocessing tools
# Set environment variables before running the app to enable auto-preprocessing:
# - OPENPOSE_BIN: full path to OpenPoseDemo executable (e.g., C:\openpose\bin\OpenPoseDemo.exe)
# - PARSER_CMD: command template to run a human parsing model. Must include
#               {input} for the input image path and {output_dir} for output folder.
#               Example: "python C:\\schp\\infer.py --image {input} --outdir {output_dir}"
OPENPOSE_BIN = os.getenv('OPENPOSE_BIN')
PARSER_CMD = os.getenv('PARSER_CMD')

# Fallback to config file if env vars are not set
CONFIG_PATH = APP_DIR / 'preprocess_config.json'
if (not OPENPOSE_BIN or not PARSER_CMD) and CONFIG_PATH.exists():
    try:
        import json as _json
        with open(CONFIG_PATH, 'r', encoding='utf-8') as _f:
            _cfg = _json.load(_f)
            OPENPOSE_BIN = OPENPOSE_BIN or _cfg.get('openpose_bin')
            PARSER_CMD = PARSER_CMD or _cfg.get('parser_cmd')
    except Exception:
        pass


def preprocess_person(base_name: str):
    """Ensure required preprocessing artifacts exist for a given person image base name.

    base_name: filename without extension, e.g., '00034_00'
    Creates (if missing):
      - image-parse/{base}.png
      - openpose-img/{base}_rendered.png
      - openpose-json/{base}_keypoints.json
    """
    created = []
    img_path = TEST_DIR / 'image' / f'{base_name}.jpg'
    parse_path = TEST_DIR / 'image-parse' / f'{base_name}.png'
    pose_img_path = TEST_DIR / 'openpose-img' / f'{base_name}_rendered.png'
    pose_json_path = TEST_DIR / 'openpose-json' / f'{base_name}_keypoints.json'

    # Run OpenPose for missing pose artifacts
    if (not pose_img_path.exists() or not pose_json_path.exists()) and OPENPOSE_BIN:
        try:
            (TEST_DIR / 'openpose-img').mkdir(parents=True, exist_ok=True)
            (TEST_DIR / 'openpose-json').mkdir(parents=True, exist_ok=True)
            cmd = [
                OPENPOSE_BIN,
                '--image_dir', str(IMG_DIR),
                '--write_json', str(TEST_DIR / 'openpose-json'),
                '--write_images', str(TEST_DIR / 'openpose-img'),
                '--render_pose', '1',
                '--display', '0'
            ]
            subprocess.run(cmd, cwd=str(Path(OPENPOSE_BIN).parent.parent), check=True, capture_output=True)
        except Exception:
            pass

        if pose_img_path.exists():
            created.append(pose_img_path.name)
        if pose_json_path.exists():
            created.append(pose_json_path.name)

    # Run human parser for missing parsing map
    if not parse_path.exists() and PARSER_CMD:
        try:
            (TEST_DIR / 'image-parse').mkdir(parents=True, exist_ok=True)
            output_dir = TEST_DIR / 'image-parse'
            cmd_str = PARSER_CMD.format(input=str(img_path), output_dir=str(output_dir))
            # Run via shell to allow complex command templates
            subprocess.run(cmd_str, shell=True, cwd=str(VITON_DIR), check=True, capture_output=True)
        except Exception:
            pass
        if parse_path.exists():
            created.append(parse_path.name)

    return created


def run_tryon_batch(people_files, clothes_files, job_prefix: str):
    job_name = f"{job_prefix}_{int(time.time())}"
    with open(DATASETS_DIR / 'test_pairs.txt', 'w', encoding='utf-8') as f:
        for p in people_files:
            for c in clothes_files:
                f.write(f"{p} {c}\n")
    (RESULTS_DIR / job_name).mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, str(VITON_DIR / 'test.py'),
        '--name', job_name,
        '--dataset_dir', str(DATASETS_DIR),
        '--checkpoint_dir', str(CHECKPOINTS_DIR),
        '--save_dir', str(RESULTS_DIR)
    ]
    proc = subprocess.run(cmd, cwd=str(VITON_DIR), capture_output=True, text=True)
    if proc.returncode != 0:
        return None, f"Inference failed:<br><pre>{proc.stdout}\n{proc.stderr}</pre>"
    return job_name, None

@app.route('/')
def index():
    people = sorted([f for f in os.listdir(IMG_DIR) if f.lower().endswith('.jpg')])
    clothes = sorted([f for f in os.listdir(CLOTH_DIR) if f.lower().endswith('.jpg')])
    return render_template('index.html', people=people, clothes=clothes)

@app.route('/preview/person/<path:filename>')
def preview_person(filename: str):
    return send_from_directory(IMG_DIR, filename)

@app.route('/preview/cloth/<path:filename>')
def preview_cloth(filename: str):
    return send_from_directory(CLOTH_DIR, filename)

@app.route('/tryon', methods=['POST'])
def tryon():
    person = request.form.get('person')
    cloth = request.form.get('cloth')
    if not person or not cloth:
        return redirect(url_for('index'))

    job_name = f"web_{int(time.time())}"

    # Ensure preprocessing exists for this person
    base = person.replace('.jpg', '')
    created = preprocess_person(base)

    # Quick existence check to fail fast with guidance
    missing = []
    if not (TEST_DIR / 'image-parse' / f'{base}.png').exists():
        missing.append(f'image-parse/{base}.png')
    if not (TEST_DIR / 'openpose-img' / f'{base}_rendered.png').exists():
        missing.append(f'openpose-img/{base}_rendered.png')
    if not (TEST_DIR / 'openpose-json' / f'{base}_keypoints.json').exists():
        missing.append(f'openpose-json/{base}_keypoints.json')
    if missing:
        msg = 'Missing preprocessing files: ' + ', '.join(missing)
        if created:
            msg += ' (created: ' + ', '.join(created) + ')'
        if not OPENPOSE_BIN or not PARSER_CMD:
            msg += '. Configure OPENPOSE_BIN/PARSER_CMD or preprocess_config.json and retry.'
        return msg, 400

    # We will reuse the dataset structure; only need a pairs file
    pairs_path = DATASETS_DIR / 'test_pairs.txt'
    with open(pairs_path, 'w', encoding='utf-8') as f:
        f.write(f"{person} {cloth}\n")

    # Ensure result subdir exists
    (RESULTS_DIR / job_name).mkdir(parents=True, exist_ok=True)

    # Invoke test.py
    cmd = [
        sys.executable, str(VITON_DIR / 'test.py'),
        '--name', job_name,
        '--dataset_dir', str(DATASETS_DIR),
        '--checkpoint_dir', str(CHECKPOINTS_DIR),
        '--save_dir', str(RESULTS_DIR)
    ]
    # Run and capture output (blocking)
    proc = subprocess.run(cmd, cwd=str(VITON_DIR), capture_output=True, text=True)
    if proc.returncode != 0:
        return f"Inference failed:<br><pre>{proc.stdout}\n{proc.stderr}</pre>", 500

    # Find produced file (pattern: <person_id>_<cloth_id>.jpg)
    out_dir = RESULTS_DIR / job_name
    generated = sorted([p.name for p in out_dir.glob('*.jpg')])
    if not generated:
        return "No result generated.", 500

    return render_template('result.html', job_name=job_name, image_name=generated[0])

@app.route('/results/<job_name>/<image_name>')
def serve_result(job_name, image_name):
    return send_from_directory(RESULTS_DIR / job_name, image_name)

@app.post('/upload_person')
def upload_person():
    f = request.files.get('person_file')
    if not f or f.filename == '':
        flash('No file selected')
        return redirect(url_for('index'))

    filename = secure_filename(f.filename)
    if not filename.lower().endswith('.jpg'):
        flash('Please upload a JPG file (e.g., 00000_00.jpg)')
        return redirect(url_for('index'))

    save_path = IMG_DIR / filename
    IMG_DIR.mkdir(parents=True, exist_ok=True)
    f.save(save_path)
    
    # Optionally autofill preprocessing from donor person
    donor = request.form.get('donor_person')
    base = filename.replace('.jpg', '')
    if donor:
        donor_base = donor.replace('.jpg', '')
        pairs = [
            (TEST_DIR / 'image-parse' / f'{donor_base}.png', TEST_DIR / 'image-parse' / f'{base}.png'),
            (TEST_DIR / 'openpose-img' / f'{donor_base}_rendered.png', TEST_DIR / 'openpose-img' / f'{base}_rendered.png'),
            (TEST_DIR / 'openpose-json' / f'{donor_base}_keypoints.json', TEST_DIR / 'openpose-json' / f'{base}_keypoints.json'),
        ]
        copied = []
        for src, dst in pairs:
            if src.exists():
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(src, dst)
                copied.append(dst.name)
        if copied:
            flash('Autofilled preprocessing from donor: ' + ', '.join(copied))

    # Check for required aux files
    created = preprocess_person(base)
    missing = []
    if not (TEST_DIR / 'image-parse' / f'{base}.png').exists():
        missing.append(f'image-parse/{base}.png')
    if not (TEST_DIR / 'openpose-img' / f'{base}_rendered.png').exists():
        missing.append(f'openpose-img/{base}_rendered.png')
    if not (TEST_DIR / 'openpose-json' / f'{base}_keypoints.json').exists():
        missing.append(f'openpose-json/{base}_keypoints.json')

    if missing:
        if created:
            flash('Some preprocessing created: ' + ', '.join(created))
        flash('Uploaded image saved, but preprocessing files are missing: ' + ', '.join(missing))
        if not OPENPOSE_BIN or not PARSER_CMD:
            flash('Auto-preprocessing is disabled. Set OPENPOSE_BIN and PARSER_CMD env vars to enable it.')
        else:
            flash('Auto-preprocessing ran but some artifacts are still missing. Check your tools.')
    else:
        msg = 'Uploaded image ready for try-on!'
        if created:
            msg += ' (created: ' + ', '.join(created) + ')'
        flash(msg)

    # Optionally auto-run try-on for this person with all clothes
    if request.form.get('auto_tryon_all') == '1':
        clothes = sorted([f for f in os.listdir(CLOTH_DIR) if f.lower().endswith('.jpg')])
        job_name, err = run_tryon_batch([filename], clothes, 'batchp')
        if err:
            flash('Batch try-on failed')
            return f"{err}", 500
        out_dir = RESULTS_DIR / job_name
        images = sorted([p.name for p in out_dir.glob('*.jpg')])
        return render_template('result.html', job_name=job_name, images=images)

    return redirect(url_for('index'))

@app.post('/upload_cloth')
def upload_cloth():
    f = request.files.get('cloth_file')
    if not f or f.filename == '':
        flash('No cloth file selected')
        return redirect(url_for('index'))

    filename = secure_filename(f.filename)
    if not filename.lower().endswith('.jpg'):
        flash('Please upload a JPG cloth image')
        return redirect(url_for('index'))

    CLOTH_DIR.mkdir(parents=True, exist_ok=True)
    cloth_path = CLOTH_DIR / filename
    f.save(cloth_path)

    # Auto-generate a crude mask using GrabCut
    try:
        img = cv2.imread(str(cloth_path))
        if img is None:
            raise RuntimeError('failed to load image')
        mask = np.zeros(img.shape[:2], np.uint8)
        rect = (5, 5, img.shape[1]-10, img.shape[0]-10)
        bgdModel = np.zeros((1, 65), np.float64)
        fgdModel = np.zeros((1, 65), np.float64)
        cv2.grabCut(img, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
        mask_bin = np.where((mask==2)|(mask==0), 0, 255).astype('uint8')
        (TEST_DIR / 'cloth-mask').mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str((TEST_DIR / 'cloth-mask' / filename)), mask_bin)
        flash('Cloth uploaded and mask generated')
    except Exception as e:
        flash(f'Cloth uploaded, but mask generation failed: {e}')

    # Optionally auto-run try-on for this cloth with all people
    if request.form.get('auto_tryon_all_people') == '1':
        people = sorted([f for f in os.listdir(IMG_DIR) if f.lower().endswith('.jpg')])
        job_name, err = run_tryon_batch(people, [filename], 'batchc')
        if err:
            flash('Batch try-on failed')
            return f"{err}", 500
        out_dir = RESULTS_DIR / job_name
        images = sorted([p.name for p in out_dir.glob('*.jpg')])
        return render_template('result.html', job_name=job_name, images=images)

    return redirect(url_for('index'))

@app.post('/tryon_person_all')
def tryon_person_all():
    person = request.form.get('person_all')
    if not person:
        return redirect(url_for('index'))
    # Ensure preprocessing exists for this person
    base = person.replace('.jpg', '')
    created = preprocess_person(base)
    missing = []
    if not (TEST_DIR / 'image-parse' / f'{base}.png').exists():
        missing.append(f'image-parse/{base}.png')
    if not (TEST_DIR / 'openpose-img' / f'{base}_rendered.png').exists():
        missing.append(f'openpose-img/{base}_rendered.png')
    if not (TEST_DIR / 'openpose-json' / f'{base}_keypoints.json').exists():
        missing.append(f'openpose-json/{base}_keypoints.json')
    if missing:
        msg = 'Missing preprocessing files: ' + ', '.join(missing)
        if created:
            msg += ' (created: ' + ', '.join(created) + ')'
        if not OPENPOSE_BIN or not PARSER_CMD:
            msg += '. Configure OPENPOSE_BIN/PARSER_CMD or preprocess_config.json and retry.'
        return msg, 400
    clothes = sorted([f for f in os.listdir(CLOTH_DIR) if f.lower().endswith('.jpg')])
    job_name = f"batchp_{int(time.time())}"
    with open(DATASETS_DIR / 'test_pairs.txt', 'w', encoding='utf-8') as f:
        for c in clothes:
            f.write(f"{person} {c}\n")
    (RESULTS_DIR / job_name).mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, str(VITON_DIR / 'test.py'),
        '--name', job_name,
        '--dataset_dir', str(DATASETS_DIR),
        '--checkpoint_dir', str(CHECKPOINTS_DIR),
        '--save_dir', str(RESULTS_DIR)
    ]
    proc = subprocess.run(cmd, cwd=str(VITON_DIR), capture_output=True, text=True)
    if proc.returncode != 0:
        return f"Inference failed:<br><pre>{proc.stdout}\n{proc.stderr}</pre>", 500
    out_dir = RESULTS_DIR / job_name
    images = sorted([p.name for p in out_dir.glob('*.jpg')])
    return render_template('result.html', job_name=job_name, images=images)

@app.post('/tryon_cloth_all')
def tryon_cloth_all():
    cloth = request.form.get('cloth_all')
    if not cloth:
        return redirect(url_for('index'))
    people = sorted([f for f in os.listdir(IMG_DIR) if f.lower().endswith('.jpg')])
    job_name = f"batchc_{int(time.time())}"
    with open(DATASETS_DIR / 'test_pairs.txt', 'w', encoding='utf-8') as f:
        for p in people:
            f.write(f"{p} {cloth}\n")
    (RESULTS_DIR / job_name).mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, str(VITON_DIR / 'test.py'),
        '--name', job_name,
        '--dataset_dir', str(DATASETS_DIR),
        '--checkpoint_dir', str(CHECKPOINTS_DIR),
        '--save_dir', str(RESULTS_DIR)
    ]
    proc = subprocess.run(cmd, cwd=str(VITON_DIR), capture_output=True, text=True)
    if proc.returncode != 0:
        return f"Inference failed:<br><pre>{proc.stdout}\n{proc.stderr}</pre>", 500
    out_dir = RESULTS_DIR / job_name
    images = sorted([p.name for p in out_dir.glob('*.jpg')])
    return render_template('result.html', job_name=job_name, images=images)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

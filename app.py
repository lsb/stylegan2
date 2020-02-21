import os
import pickle
import PIL.Image
import numpy as np
import dnnlib
import dnnlib.tflib as tflib
import pretrained_networks
from encoder.generator_model import Generator
from encoder.perceptual_model import PerceptualModel
from flask import Flask, request, abort, send_file
import base64
import io
import time
from collections import namedtuple
import tensorflow as tf
from ffhq_dataset.face_alignment import image_align
from ffhq_dataset.landmarks_detector import LandmarksDetector
import tempfile

model_url = "gdrive:networks/stylegan2-ffhq-config-f.pkl"
tflib.init_tf()
generator_network, discriminator_network, Gs_network = pretrained_networks.load_networks(model_url)
generator = Generator(Gs_network, 1, randomize_noise=False)

app = Flask(__name__)

def u64latents_to_latents(u64latents):
    if len(u64latents) != 1 * 18 * 512 * 4 * 4 / 3:
        return None
    return np.ndarray((1,18,512), dtype=np.float32, buffer=base64.urlsafe_b64decode(u64latents))

def dict_as_namedtuple(d, name=''):
    return namedtuple(name, [name for name in d])(**d)

landmarks_path = "shape_predictor_68_face_landmarks.dat"
landmarks_detector = LandmarksDetector(landmarks_path)

@app.route('/alignfaces', methods=['GET'])
def alignfaces():
    u64rawfaces = request.args.get('u64rawfaces')
    if len(u64rawfaces) > 16 * 1024 * 1024:
        abort(413)
    retval = None
    with tempfile.NamedTemporaryFile(dir=".") as raw_input:
        PIL.Image.open(io.BytesIO(base64.urlsafe_b64decode(u64rawfaces))).save(raw_input.name, 'jpeg', quality=90)
        face_landmarks = [x for x in landmarks_detector.get_landmarks(raw_input.name)]
        faces = [io.BytesIO() for f in face_landmarks]
        for i, facepng in enumerate(faces):
            image_align(raw_input.name, facepng, face_landmarks[i])
            facepng.seek(0)
            webp = io.BytesIO()
            PIL.Image.open(facepng).save(webp, "webp")
            webp.seek(0)
            faces[i] = webp.getvalue()
        faces.sort(key=len, reverse=True)
        retval = b"\n".join([base64.urlsafe_b64encode(f) for f in faces])
    return retval



@app.route('/render9000', methods=['GET'])
def render9000():
    latents = u64latents_to_latents(request.args.get('u64latents', ''))
    if latents is None:
        abort(413)
    generator.set_dlatents(latents)
    img = PIL.Image.fromarray(generator.generate_images()[0], 'RGB')
    webp = io.BytesIO()
    img.save(webp, "webp")
    webp.seek(0)
    return send_file(webp, "image/webp")

@app.route('/optimize', methods=['GET'])
def optimize():
    latents = u64latents_to_latents(request.args.get('u64latents'))
    if latents is None:
        abort(413)
    u64reference_webp = request.args.get('u64reference')
    reference_image = io.BytesIO(base64.urlsafe_b64decode(u64reference_webp)) # todo: restrict input formats
    generator.set_dlatents(latents)
    model_url = "gdrive:networks/stylegan2-ffhq-config-f.pkl"
    vgg_url = "https://d36zk2xti64re0.cloudfront.net/stylegan1/networks/metrics/vgg16_zhang_perceptual.pkl"
    optional_args = {
        "lr": 0.857,
        "decay_rate": 0.95,
        "iterations": 25,
        "decay_steps": 4,
        "image_size": 256,
        "use_vgg_layer": 9,
        "use_vgg_loss": 100,
        "use_pixel_loss": 1,
        "use_mssim_loss": 100,
        "use_lpips_loss": 0,
        "use_l1_penalty": 1,
        "use_adaptive_loss": False, # requires tf >= 2 😿
        "sharpen_input": False,
        "batch_size": 1,
        "use_discriminator_loss": 0,
        "optimizer": "ggt",
        "average_best_loss": 0.25,
    }
    forced_args = {
        "face_mask": False,
        "use_grabcut": None,
        "scale_mask": None,
        "mask_dir": None,
        "vgg_url": vgg_url, # we do not load random pickles that Internet Users ask us to
        "batch_size": 1,
        "model_url": model_url, # we do not load random pickles that Internet Users ask us to
        "model_res": 1024,
    }
    # print(f"{time.time()} default config done")
    merged_args = {k: type(optional_args.get(k))(request.args.get(k,optional_args.get(k))) for k in optional_args}
    for k in forced_args: merged_args[k] = forced_args[k]
    args = dict_as_namedtuple(merged_args, name="args")
    # print(f"{time.time()} config merged")

    perc_model = None
    if (args.use_lpips_loss > 0.00000001):
        with dnnlib.util.open_url(args.vgg_url, cache_dir='.stylegan2-cache') as f:
            perc_model = pickle.load(f)

    # print(f"{time.time()} perc model")
    perceptual_model = PerceptualModel(args, perc_model=perc_model, batch_size=args.batch_size)
    # print(f"{time.time()} perceptual model object")
    with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):
        perceptual_model.build_perceptual_model(generator, discriminator_network)
    # print(f"{time.time()} reused scope built")

    perceptual_model.set_reference_images([reference_image])
    # print(f"{time.time()} refernce images set")
    op = perceptual_model.optimize(generator.dlatent_variable, iterations=args.iterations, use_optimizer=args.optimizer)
    # print(f"{time.time()} optimizer iterable created")
    best_loss = None
    best_dlatent = None
    for loss_dict in op:
        if best_loss is None or loss_dict["loss"] < best_loss:
            if best_dlatent is None:
                best_dlatent = generator.get_dlatents()
            else:
                best_dlatent = args.average_best_loss * best_dlatent + (1 - args.average_best_loss) * generator.get_dlatents()
            generator.set_dlatents(best_dlatent)
            best_loss = loss_dict["loss"]
        generator.stochastic_clip_dlatents()
        # print(f"best loss: {best_loss} @ {time.time()}")

    return base64.urlsafe_b64encode(best_dlatent.tobytes('C'))

if __name__ == "__main__":
    app.run(threaded=False, processes=1)

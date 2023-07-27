from .utils import getlogger, untuple
import comfy.sample
import sys

log = getlogger()

# AITemplate support
def get_aitemplate_module():
    return sys.modules['AIT.AITemplate.AITemplate']


def get_callback(model):
    if isinstance(model, tuple):
        model = model[0]
    return getattr(model, 'prompt_control_callback', None)

def do_hijack():
    orig_sampler = comfy.sample.sample
    if hasattr(comfy.sample.sample, 'pc_hijack_done'):
        return
    def pc_sample(*args, **kwargs):
        model = args[0]
        cb = get_callback(model)
        if cb:
            return cb(orig_sampler, *args, **kwargs)
        else:
            return orig_sampler(*args, **kwargs)
    comfy.sample.sample = pc_sample

    try:
        global aitemplate_mod
        log.info("AITemplate detected, hijacking...")
        AITLoader = sys.modules['AIT.AITemplate.ait.load'].AITLoader
        orig_apply_unet = AITLoader.apply_unet
        def apply_unet(self, *args, **kwargs):
            setattr(self, 'pc_applied_module', kwargs['aitemplate_module'])
            return orig_apply_unet(self, *args, **kwargs)
        AITLoader.apply_unet = apply_unet
        log.info("AITemplate hijack complete")
    except Exception as e:
        log.info("AITemplate hijack failed or not necessary")

    setattr(comfy.sample.sample, 'pc_hijack_done', True)
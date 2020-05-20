from .saliency import get_image_saliency_result, get_image_saliency_plot

__all__ = ['getCAM']


def getCAM(model, raw_image, input, label, layer_path=None, display=True, save_path=False):
    saliency_maps = get_image_saliency_result(model, raw_image, input, label,
                                              methods=['gradcam'], layer_path=layer_path)
    figure = get_image_saliency_plot(saliency_maps, display=display, save_path=save_path)

    return figure

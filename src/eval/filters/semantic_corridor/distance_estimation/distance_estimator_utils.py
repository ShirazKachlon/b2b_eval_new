import numpy as np 

CAMERA_MODEL = {'focus_distnace':314,
                'fixed_dims_height':{'ped':1.7, 'car':1.45, 'van':2, 'truck':4}}

def longitudinal_distnace(x, label):
    focus_distnace = CAMERA_MODEL['focus_distnace']
    fixed_dims_height = CAMERA_MODEL['fixed_dims_height']
    if label==0:
        return focus_distnace * fixed_dims_height['ped'] / x
    elif label==2001:
        return focus_distnace * fixed_dims_height['car'] / x
    elif label==2003:
        return focus_distnace * fixed_dims_height['van'] / x
    elif label==2004:
        return focus_distnace * fixed_dims_height['truck'] / x

def x_to_angle(xc,img_width=3840, fov=120):
    """_summary_
        The function returns the angle of an object in radians in relation to image center

    Args:
        xc: objects x center vector
        img_width: image width. Defaults to 3840.
        fov: camera field of view. Defaults to 120.

    Returns:
        objects angle vector
    """
    fov_rad = fov*np.pi/180
    x = [0, img_width/2, img_width]
    y = [-fov_rad/2,0,fov_rad/2]
    coeff = np.polyfit(x, y, 1)
    return xc*coeff[0] + coeff[1]
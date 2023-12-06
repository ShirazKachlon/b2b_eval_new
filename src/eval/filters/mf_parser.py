import pandas as pd
import cv2
import os

# TODO: refactor this file to be more readable
def validate_bbox(left, top, width, height, image_width, image_height) -> bool:
    """ returns whether a bbox is inside a box """
    if any(v < 0 for v in (left, top, width, height)):
        return False
    if any(v > image_width for v in (left, width, left + width)):
        return False
    if any(v > image_height for v in (top, height)):
        return False
    return True


def find_y_at_x(x1, x2, y1, y2, x_target):
    """
    Calculate the y-coordinate where x = x_target for a line passing through points (x1, y1) and (x2, y2).
    Returns: y-coordinate where x = x_target.
    """
    slope = (y2 - y1) / (x2 - x1 + 1e-16)
    y_at_x = y1 + slope * (x_target - x1)
    return y_at_x


def default_truncation(row, im_height):
    y_st = min(max(row.y_st, 0), im_height - 1)
    y_rt = min(max(row.y_rt, 0), im_height - 1)
    y_rb = min(max(row.y_rb, 0), im_height - 1)
    y_sb = min(max(row.y_sb, 0), im_height - 1)
    y_lb = min(max(row.y_lb, 0), im_height - 1)
    y_lt = min(max(row.y_lt, 0), im_height - 1)
    top = min([y_st, y_lt, y_rt])
    bottom = max([y_rb, y_sb, y_lb])
    return top, bottom


def truncate(row, im_width, im_height):
    # two cases where the left of the BB is visible
    if row.x_l > 0:
        if row.x_r >= im_width and row.x_s >= im_width:
            y_0b = find_y_at_x(row.x_l, row.x_s, row.y_lb, row.y_sb, x_target=im_width)
            bottom = max([y_0b, row.y_lb])  # y_sb is not visible

            y_0t = find_y_at_x(row.x_l, row.x_s, row.y_lt, row.y_st, x_target=im_width)
            top = min([y_0t, row.y_lt])  # y_st is not visible

        if row.x_r >= im_width and row.x_s < im_width:
            y_0b = find_y_at_x(row.x_r, row.x_s, row.y_sb, row.y_rb, x_target=im_width)
            bottom = max([y_0b, row.y_lb, row.y_sb])

            y_0t = find_y_at_x(row.x_r, row.x_s, row.y_st, row.y_rt, x_target=im_width)
            top = min([y_0t, row.y_lt, row.y_st])

    # two cases where only the right of the BB is visible 
    else:
        if row.x_s >= 0 and row.x_r >= 0:
            y_0b = find_y_at_x(row.x_l, row.x_s, row.y_lb, row.y_sb, x_target=0)
            bottom = max([y_0b, row.y_rb, row.y_sb])

            y_0t = find_y_at_x(row.x_l, row.x_s, row.y_lt, row.y_st, x_target=0)
            top = min([y_0t, row.y_rt, row.y_st])

        if row.x_s < 0 and row.x_r >= 0:
            y_0b = find_y_at_x(row.x_s, row.x_r, row.y_sb, row.y_rb, x_target=0)
            bottom = max([y_0b, row.y_rb])  # y_sb is not visible

            y_0t = find_y_at_x(row.x_s, row.x_r, row.y_st, row.y_rt, x_target=0)
            top = min([y_0t, row.y_rt])  # y_sb is not visible

    # (bb is inside image) or (weird case where x_l=x_s)
    if (row.x_l > 0 and row.x_r < im_width) or (row.x_s == row.x_l):
        top, bottom = default_truncation(row, im_height)

    # for other edge cases, just make sure top and bottom are valid
    if top < 0:
        top = 0
    if bottom >= im_height:
        bottom = im_height - 1
    return top, bottom


def convert_3d_2d(tsv_path, im_width=3840.0, im_height=1920.0, images_path=None, images_out_path=None):
    if images_out_path and not os.path.exists(images_out_path):
        os.makedirs(images_out_path)

    df = pd.read_csv(tsv_path, sep="\t")
    for index, row in df.iterrows():
        if index % 1000 == 0:
            print(f'running on row {index} / {len(df)}')

        top = min([row.y_lt, row.y_rt, row.y_st])
        left = row.x_l

        valid = validate_bbox(left, top, row.width, row.height, im_width, im_height)
        if not valid:

            # truncate coordinates
            x_l = min(max(row.x_l, 0), im_width - 1)
            x_r = min(max(row.x_r, 0), im_width - 1)

            top, bottom = truncate(row, im_width, im_height)

            x_center = (x_l + x_r) / 2
            y_center = (bottom + top) / 2
            width = x_r - x_l
            height = bottom - top

            # assign to dataframe
            df.at[index, 'x_center'] = x_center
            df.at[index, 'y_center'] = y_center
            df.at[index, 'width'] = width
            df.at[index, 'height'] = height

            if images_path:
                image_path = os.path.join(images_path, str(row['name']) + '.png')

                if not os.path.exists(image_path):
                    continue

                image_out_path = os.path.join(images_out_path, str(row['name']) + '.png')
                cv_image = cv2.imread(image_path)

                lines = []
                lines.append((int(x_l), int(top), int(x_r), int(top)))  # (top left, top right)
                lines.append((int(x_l), int(bottom), int(x_r), int(bottom)))  # (bottom left, bottom right)
                lines.append((int(x_l), int(top), int(x_l), int(bottom)))  # (top left, bottom left)
                lines.append((int(x_r), int(top), int(x_r), int(bottom)))  # (top right, bottom right)

                for line in lines:
                    cv_image = cv2.line(cv_image, line[:2], line[2:], (0, 0, 255), 4)
                cv2.imwrite(image_out_path, cv_image)

    return df
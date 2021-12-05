# function to convert base64 string to image array python
def base64_to_image(base64_string):
    # decode base64 string to image
    img = Image.open(BytesIO(base64.b64decode(base64_string)))
    # convert image to array
    img_array = np.array(img)
    # return array
    return img_array
import cv2

face_classifier = cv2.CascadeClassifier(
    "cascade/haarcascade_frontalface_default.xml")
nariz_classifier = cv2.CascadeClassifier("cascade/haarcascade_mcs_nose.xml")

mustache = cv2.imread("img/mustache.png", cv2.IMREAD_UNCHANGED)
cont = 0


def image_resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]
    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image
    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)
    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation=inter)
    # return the resized image
    return resized


def filtro_linear(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces_return = face_classifier.detectMultiScale(img_gray, 1.2, 5)

    for (x, y, w, h) in faces_return:
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
        roi_color = img[y:y+h, x:x+w]
        blur = cv2.GaussianBlur(roi_color, (101, 101), 0)
        img[y:y+h, x:x+w] = blur

    return img


def filtro_sobreposicao(img):

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    faces_return = face_classifier.detectMultiScale(img_gray, 1.2, 4)

    for (x, y, w, h) in faces_return:
        roi_gray = img_gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        nose = nariz_classifier.detectMultiScale(roi_gray, 1.2, 4)
        for (nx, ny, nw, nh) in nose:
            #cv2.rectangle(roi_color, (nx, ny), (nx + nw, ny + nh), (255, 0, 0), 3)
            roi_nose = roi_gray[ny: ny + nh, nx: nx + nw]
            mustache2 = image_resize(mustache.copy(), width=nw)

            mw, mh, mc = mustache2.shape
            for i in range(0, mw):
                for j in range(0, mh):
                    # print(glasses[i, j]) #RGBA
                    if mustache2[i, j][3] != 0:  # alpha 0
                        roi_color[ny + int(nh/1.5) + i, nx +
                                  j] = mustache2[i, j]

    return img


def mouse_click(event, x, y, flags, param):
    global cont
    if event == cv2.EVENT_LBUTTONDOWN:
        cont += 1
        while(True):
            rval, img = cap.read()
            if cont % 2 == 0:
                frame = filtro_sobreposicao(img)
                cv2.imshow('NAC02', frame)
                cv2.setMouseCallback('NAC02', mouse_click)
                cv2.waitKey(30)

            else:
                frame = filtro_linear(img)
                cv2.imshow('NAC02', frame)
                cv2.setMouseCallback('NAC02', mouse_click)
                cv2.waitKey(30)

    elif event == cv2.EVENT_RBUTTONDOWN:
        while(True):
            rval, img = cap.read()
            cv2.imshow('NAC02', img)
            cv2.setMouseCallback('NAC02', mouse_click)
            cv2.waitKey(30)


cap = cv2.VideoCapture('video.mp4')

while(cap.isOpened()):
    rval, img = cap.read()

    cv2.imshow('NAC02', img)
    cv2.setMouseCallback('NAC02', mouse_click)
    cv2.waitKey(30)

cv2.destroyAllWindows()

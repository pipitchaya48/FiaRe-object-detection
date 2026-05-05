from Export_modules.detectAndLabel import detectAndLabel
import cv2

root = "tests"

test_imgs = ["test5.jpg", "test2.webp", "test3.webp", "test4.jpg", "test6.webp", "test7.jpeg"]

# for test_img in test_imgs:
#     img = cv2.imread(root + "/" + test_img)
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     cv2.imwrite(root + "/gray_"+test_img, gray)

# for test_img in test_imgs:
#     detectAndLabel(root+"/gray_"+test_img)

for test_img in test_imgs:
    detectAndLabel(root+"/"+test_img)

# img = cv2.imread("tests/test7.jpeg")
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# cv2.imwrite("tests/gray_test6.jpeg", gray)
detectAndLabel("tests/test.jpg")





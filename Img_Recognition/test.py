from Export_modules.detectAndLabel import detectAndLabel

test_img_path = "car_detection/test/test2.webp"

ret = detectAndLabel(test_img_path)

print(ret)
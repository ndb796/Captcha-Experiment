from captcha.image import ImageCaptcha
import itertools
import os
import shutil
import uuid

import captcha_setting as setting


def generate_captcha(directory, characters, n):
    if os.path.exists(directory):
        shutil.rmtree(directory)
    if not os.path.exists(directory):
        os.makedirs(directory)

    image = ImageCaptcha(width=setting.width, height=setting.height, fonts=['./fonts/NanumGothicBold.ttf'])
    for count in range(1, n + 1):
        print('Generating: (' + str(count) + '/' + str(n) + ')')
        for i in itertools.permutations(characters, setting.number_per_image):
            captcha = ''.join(i)
            file_name = directory + '/' + captcha + '_' + str(uuid.uuid4()) + '.png'
            image.write(captcha, file_name)


# Execute
generate_captcha(setting.train_dataset_path, setting.character_set, 3)
generate_captcha(setting.test_dataset_path, setting.character_set, 1)

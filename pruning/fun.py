import os

def count_files():
    home = '/home/gridsan/groups/ImageNet_shared/ILSVRC2012/train/'
    total_images = sum([len(os.listdir(home + synset)) for synset in os.listdir(home)])
    print(total_images)

count_files()
        

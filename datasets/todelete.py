import os
import shutil

print(os.getcwd())

images = 'images/'
labels = 'labels/'
#
# # os.mkdir("train")
# os.mkdir("valid")
# # os.mkdir("test")
# #
# # os.mkdir("train/"+images)
# # os.mkdir("train/"+labels)
# os.mkdir("valid/"+images)
# os.mkdir("valid/"+labels)
# # os.mkdir("test/"+images)
# # os.mkdir("test/"+labels)
#
# with open('train.txt') as trainf:
#     for line in trainf.readlines():
#         try:
#             shutil.copyfile(images+line.strip(), "train/"+images+line.strip())
#             shutil.copyfile(labels+line.strip().replace(".jpg",".txt"), "train/"+labels+line.strip().replace(".jpg",".txt"))
#         except:
#             continue
#
# with open('valid.txt') as trainf:
#     for line in trainf.readlines():
#         try:
#             shutil.copyfile(images+line.strip(), "valid/"+images+line.strip())
#             shutil.copyfile(labels+line.strip().replace(".jpg",".txt"), "valid/"+labels+line.strip().replace(".jpg",".txt"))
#         except:
#             continue
# with open('test.txt') as trainf:
#     for line in trainf.readlines():
#         try:
#             shutil.copyfile(images+line.strip(), "test/"+images+line.strip())
#             shutil.copyfile(labels+line.strip().replace(".jpg",".txt"), "test/"+labels+line.strip().replace(".jpg",".txt"))
#         except:
#             continue
#
# #
# #
# #
# #
# #

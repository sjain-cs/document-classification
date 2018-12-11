Dependencies:
torchvision
torch
Pillow
Numpy

Download and unzip data folder inside this directory:
Link: https://drive.google.com/open?id=1Rc4cifQOJ6FaEMdGw5s_JHTxT4V5mKnb

Download and unzip models folder inside this directory:
Link: https://drive.google.com/open?id=1ZvdETE9LsmW_Nnh4VolC6IBjpezpsp4F

For testing using pre-trained model:
python test.py data/document_classification/val/invoice/2028701183.jpg

For computing validation stats on pretrained model
python validate.py

To re-train the model:
python train.py

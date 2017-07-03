"""Test the network."""

#import _init_paths
import pdb 
import numpy as np
import os
from caffe.proto import caffe_pb2
import sys
import argparse
import glob
import cv2
from operator import itemgetter
from caffe import io
import random
import caffe


def parse_args():
    parser = argparse.ArgumentParser()
    # Required arguments: input and output files.
    parser.add_argument(
        "--input_file_dir",dest ='input_file_dir',
        help="Input image pairs directory"
    )

    # Optional arguments.
    parser.add_argument(
        "--model_def",dest='model_def',
        default= "/deploy.prototxt",
        help="Model definition file."
    )
    parser.add_argument(
        "--pretrained_model",dest='model_weights',
        help="Trained model weights file."
    )
    parser.add_argument(
        "--binary_mean",dest='binary_mean_file',
        help="model binary mean file."
    )

    if(len(sys.argv)<2):
        parser.print_help()
        return None

    args = parser.parse_args()
    return args

class  FaceNetExtractor:
    def __init__(self, model_def, model_weights,binary_mean_file):
        caffe.set_mode_cpu()
        self.blob_size_high = 144
        self.blob_size_wide = 56

        self.load_mean(binary_mean_file)
        self.net = caffe.Net(model_def,      # defines the structure of the model
                model_weights,  # contains the trained weights
                caffe.TEST)     # use test mode (e.g., don't perform dropout)

    # convert between .binaryproto to .npy
    def load_mean(self,binary_mean_file):
       
        blob = caffe.proto.caffe_pb2.BlobProto()
        data = open(binary_mean_file,'rb').read()
        blob.ParseFromString(data)
        self.pixel_mean = io.blobproto_to_array(blob).mean(1)
       

    def prep_im_for_blob(self, im):
        target_size_high=self.blob_size_high
        target_size_wide=self.blob_size_wide 
        #pixel_means = np.array([[[102.9801, 115.9465, 122.7717]]])
        im = im.astype(np.float32, copy=False)
        im = cv2.resize(im, (target_size_wide ,target_size_high),
                    interpolation=cv2.INTER_LINEAR)
        #print('{}-{}'.format(im.shape, self.pixel_mean.shape))
        im_mean = self.pixel_mean.transpose((1,2,0))
        im_mean = cv2.resize(im_mean, (target_size_wide ,target_size_high))
        #pdb.set_trace()
        im_mean = [im_mean]
        im_mean = np.transpose(im_mean,(1,2,0))
        #im_mean = self.pixel_mean.transpose((1,2,0))
        im -= im_mean

        return im
    def extract_feature(self, img_file):
        img = cv2.imread(img_file)
        img  = self.prep_im_for_blob(img)
        blob = img[np.newaxis,...]
        channel_swap = (0, 3, 1, 2)
        blob = blob.transpose(channel_swap)
        self.net.blobs['data'].data[...] = blob

        ### perform classification
        self.net.forward()
        feat = self.net.blobs['fc7'].data[0,...]
        #print('feat mean: {}'.format(np.mean(feat)))
        return  feat.flatten()
    def batch_extract_feature(self, img_files_list):
        batch_blobs = np.zeros((len(img_files_list),3,self.blob_size,self.blob_size))
        for img_idx, img_file in enumerate(img_files_list):
            img = cv2.imread(img_file)
            img  = self.prep_im_for_blob(img)
            blob = img[np.newaxis,...]
            channel_swap = (0, 3, 1, 2)
            blob = blob.transpose(channel_swap)
            batch_blobs[img_idx,:,:,:] = blob 
        self.net.blobs['data'].reshape(*batch_blobs.shape)
        self.net.blobs['data'].data[...] = batch_blobs

        ### perform classification
        self.net.forward()
        feat = self.net.blobs['norml3'].data[...]
        return  feat.reshape((len(img_files_list),-1))

    def compute_score(self,feat_1,feat_2):
        #print('feat dif mean: {}'.format(np.mean(feat_1 - feat_2)))
        #dif = feat_1 -feat_2
        #dist =  np.sqrt(np.dot(dif,dif))
        dist = 1 - np.dot(feat_1,feat_2)
        return  dist
    def process_image_pair(self,img_file_1, img_file_2):
        img_feat_1 = self.extract_feature(img_file_1)
        img_feat_2 = self.extract_feature(img_file_2)
        distance = self.compute_score(img_feat_1, img_feat_2)
        return distance




def select_camera_samples(camera_dir,only_one= False):
    person_list = os.listdir(camera_dir)
    query_img_list =[]
    for person_id in person_list:
        person_imgs_dir = os.path.join(camera_dir, person_id)
        imgs_list = glob.glob(os.path.join(person_imgs_dir,'*.jpg'))
        if only_one:
            index = random.randint(0,len(imgs_list)-1)
            query_img_list.append(imgs_list[index])
        else:
            query_img_list.extend(imgs_list)
    return  query_img_list


if __name__ == '__main__':
    
    #temporal directory 
    tmp_dir ='/home/nikoong/Algorithm_test/dgd_person_reid/external/caffe/temp/'
    #The caffe path
    netdir = '/home/nikoong/Algorithm_test/dgd_person_reid/external/caffe/'
    #Image path used for testing
    indir = '/home/nikoong/Algorithm_test/New_dataset/'
    
    model_def   = os.path.join(netdir,'Pretrained_models/jstl_dgd_deploy_inference.prototxt')
    model_weights = os.path.join(netdir,'Pretrained_models/jstl_dgd_inference.caffemodel');
    model_mean_file = os.path.join(indir,'mean.binaryproto')

    feat_lens = 256
    caffe.set_mode_gpu()
    extractor = FaceNetExtractor(model_def,model_weights,model_mean_file)
    # Parameters

    test_num=100
    row_num = 5
    col_num = 10
    rank_num = row_num*col_num

    #cama_imgs_list = glob.glob(os.path.join(indir,'val/CamA_*_01.jpg'))
    #cama_imgs_list.extend(glob.glob(os.path.join(indir,'test0/CamA_*_01.jpg')))
    cama_imgs_list = glob.glob(os.path.join(indir,'cam_0/*.bmp'))
    #cama_imgs_list = select_camera_samples(os.path.join(indir,'1'),True)
    cama_feat_mats = np.zeros((len(cama_imgs_list),feat_lens))
    dump_cama_feat_file = os.path.join(tmp_dir,'cama_feat.npy')

if(os.path.exists(dump_cama_feat_file)):
    print('load cama feature:{}'.format(dump_cama_feat_file))
    cama_feat_mats = np.load(dump_cama_feat_file)
else:
    print('cama:extract feature,total {}:...'.format(len(cama_imgs_list)))
    for img_idx, cama_img_file in enumerate(cama_imgs_list):
        cama_feat_mats[img_idx,:] = extractor.extract_feature(cama_img_file)   
    np.save(dump_cama_feat_file,cama_feat_mats)


print extractor.extract_feature(cama_imgs_list[0]) 
print np.shape(extractor.extract_feature(cama_imgs_list[0]) )

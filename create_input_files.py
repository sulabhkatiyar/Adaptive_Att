from utils import create_input_files
import os

# Do you want to delete the previous files which may have different minimum word frequency and other attributes?
delete_previous_files = True


if __name__ == '__main__':
    if delete_previous_files == True:
        for file_name in os.listdir('path_to_data_files'):
            if 'flickr8k' in file_name.split('_'):
                os.remove('path_to_data_files' + '/' + file_name)

    # Create input files (along with word map)
    create_input_files(dataset='flickr8k',
                       karpathy_json_path='path_to_karpathy_split' + '/karpathy_split/dataset_flickr8k.json',
                       image_folder= 'path_to_raw_dataset' + '/f8k/flickr_data/Flickr_Data/Images',
                       captions_per_image=5,
                       min_word_freq=5,
                       output_folder='path_to_data_files',
                       max_len=34)

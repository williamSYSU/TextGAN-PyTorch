import os
import torch
import torchtext

from instructor.real_data.instructor import BasicInstructor
from utils.text_process import tokenize
from utils.data_loader import DataSupplier


# TO DO:
# 1. train embedding if not exists (if oracle, then always retrain )
# 2. create data generator (categorical and non categorical) (based on given dataset)
# 3. create disc and gen
# 4. train epochs and each 10 epochs print metrics
# 5. show metrics
# 6. save? or save each 10 epochs

# chack target real/fake to be right (Uniform or const)


class FixemGANInstructor(BasicInstructor):
    def __init__(self, cfg):
        super(FixemGANInstructor, self).__init__(cfg)
        # check if embeddings already exist for current oracle
        if os.path.exists(f'oracle path/{cfg.embedding_file_name}'):
            # train embedding with oracle
        w2v = load_embedding(f'oracle path/{cfg.embedding_file_name}')

        print(self.train_data)

        print(self.train_data_list)

        # data_generator = DataSupplier

        DataLoader(
            dataset=GANDataset(),
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            drop_last=True
        )


        # try:
        #     self.train_data = GenDataIter(cfg.train_data)
        #     self.test_data = GenDataIter(cfg.test_data, if_test_data=True)
        # except:
        #     pass

        # try:
        #     self.train_data_list = [GenDataIter(cfg.cat_train_data.format(i)) for i in range(cfg.k_label)]
        #     self.test_data_list = [GenDataIter(cfg.cat_test_data.format(i), if_test_data=True) for i in
        #                            range(cfg.k_label)]
        #     self.clas_data_list = [GenDataIter(cfg.cat_test_data.format(str(i)), if_test_data=True) for i in
        #                            range(cfg.k_label)]

        #     self.train_samples_list = [self.train_data_list[i].target for i in range(cfg.k_label)]
        #     self.clas_samples_list = [self.clas_data_list[i].target for i in range(cfg.k_label)]
        # except:
        #     pass



    def one_more_batch_for_generator(
        self, generator_acc, leave_in_generator_min=0.1, leave_in_generator_max=0.9
    ):
        generator_acc = min(leave_in_generator_max, generator_acc)
        generator_acc = max(leave_in_generator_min, generator_acc)
        if random.random() > generator_acc:
            return True
        return False

    def write_txt_file(self, source, save_path, save_filename):
        with open(os.path.join(save_path, save_filename), 'w') as f:
            for _, text in source:
                line = ' '.join(tokenize(text))
                f.write(line)
                f.write('\n')

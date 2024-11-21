import unittest
import numpy as np
import pandas as pd
from unittest.mock import MagicMock

from preprocessing.preprocessing import utils


class TestBaseTextCategorizationDataset(unittest.TestCase):
    def test__get_num_train_samples(self):
        """
        we want to test the class BaseTextCategorizationDataset
        we use a mock which will return a value for the not implemented methods
        then with this mocked value, we can test other methods
        """
        base = utils.BaseTextCategorizationDataset(20, 0.8)
        base._get_num_samples = MagicMock(return_value=100)
        self.assertEqual(base._get_num_train_samples(), 80)

    def test__get_num_train_batches(self):
        """
        same idea as what we did to test _get_num_train_samples
        """
        base = utils.BaseTextCategorizationDataset(20, 0.8)
        base._get_num_samples = MagicMock(return_value=100)
        self.assertEqual(base._get_num_train_batches(), 4)

    def test__get_num_test_batches(self):

        base = utils.BaseTextCategorizationDataset(20, 0.8)
        base._get_num_samples = MagicMock(return_value=100)
        self.assertEqual(base._get_num_test_batches(), 1)

    def test_get_index_to_label_map(self):
        base = utils.BaseTextCategorizationDataset(20, 0.8)
        base._get_label_list = MagicMock(return_value=['label_a', 'label_b'])
        self.assertEqual(base.get_index_to_label_map(),
                         {0: 'label_a', 1: 'label_b'})

    def test_index_to_label_and_label_to_index_are_identity(self):
        base = utils.BaseTextCategorizationDataset(20, 0.8)
        base._get_label_list = MagicMock(return_value=['label_a', 'label_b'])
        index_to_label = base.get_index_to_label_map()
        label_to_index = base.get_label_to_index_map()
        self.assertEqual(index_to_label, {
                         v: k for k, v in label_to_index.items()})

    def test_to_indexes(self):
        base = utils.BaseTextCategorizationDataset(20, 0.8)
        base._get_label_list = MagicMock(return_value=['label_a', 'label_b'])
        self.assertEqual(base.to_indexes(['label_a', 'label_b']), [0, 1])


class TestLocalTextCategorizationDataset(unittest.TestCase):
    def test_load_dataset_returns_expected_data(self):
        # we mock pandas read_csv to return a fixed dataframe
        pd.read_csv = MagicMock(return_value=pd.DataFrame({
            'post_id': ['id_1', 'id_2'],
            'tag_name': ['tag_a', 'tag_b'],
            'tag_id': [1, 2],
            'tag_position': [0, 1],
            'title': ['title_1', 'title_2']
        }))
        dataset = utils.LocalTextCategorizationDataset.load_dataset(
            "fake_path", 1)
        expected = pd.DataFrame({
            'post_id': ['id_1'],
            'tag_name': ['tag_a'],
            'tag_id': [1],
            'tag_position': [0],
            'title': ['title_1']
        })
        pd.testing.assert_frame_equal(dataset, expected)

    def test_get_num_train_batches_less_than_zero(self):
        pd.read_csv = MagicMock(return_value=pd.DataFrame({
            'post_id': ['id_1', 'id_2', 'id_3', 'idç_4', 'id_5', 'id_6', 'id_7', 'id_8', 'id_9', 'id_10'],
            'tag_name': ['tag_a', 'tag_b', 'tag_c', 'tag_b', 'tag_a', 'tag_b', 'tag_a', 'tag_b', 'tag_a', 'tag_b'],
            'tag_id': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            'tag_position': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            'title': ['title_1', 'title_2', 'title_3', 'title_4', 'title_5', 'title_6', 'title_7', 'title_8', 'title_9', 'title_10']
        }))
        # We mock a dataset with 10 samples and we want a batch size of 11
        # so we should get an assertion error just by trying to create batches for the train set

        with self.assertRaises(AssertionError):
            utils.LocalTextCategorizationDataset(
                "fake_path", batch_size=11, min_samples_per_label=1)

    def test_get_num_test_batches_less_than_zero(self):
        pd.read_csv = MagicMock(return_value=pd.DataFrame({
            'post_id': ['id_1', 'id_2', 'id_3', 'idç_4', 'id_5', 'id_6', 'id_7', 'id_8', 'id_9', 'id_10'],
            'tag_name': ['tag_a', 'tag_b', 'tag_c', 'tag_b', 'tag_a', 'tag_b', 'tag_a', 'tag_b', 'tag_a', 'tag_b'],
            'tag_id': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            'tag_position': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            'title': ['title_1', 'title_2', 'title_3', 'title_4', 'title_5', 'title_6', 'title_7', 'title_8', 'title_9', 'title_10']
        }))
        # Here we want a batch size of 3 and we have 10 samples.
        # Train set will have 8 samples and test set will have 2 samples.
        # So we should get an assertion error by trying to create batches for the test set
        with self.assertRaises(AssertionError):
            utils.LocalTextCategorizationDataset(
                "fake_path", batch_size=3, min_samples_per_label=1)

    def test__get_label_list_returns_expected_data(self):
        pd.read_csv = MagicMock(return_value=pd.DataFrame({
            'post_id': ['id_1', 'id_2', 'id_3', 'idç_4', 'id_5', 'id_6', 'id_7', 'id_8', 'id_9', 'id_10'],
            'tag_name': ['tag_a', 'tag_b', 'tag_c', 'tag_b', 'tag_a', 'tag_b', 'tag_a', 'tag_b', 'tag_a', 'tag_b'],
            'tag_id': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            'tag_position': [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
            'title': ['title_1', 'title_2', 'title_3', 'title_4', 'title_5', 'title_6', 'title_7', 'title_8', 'title_9', 'title_10']
        }))
        dataset = utils.LocalTextCategorizationDataset(
            "fake_path", batch_size=2, min_samples_per_label=1)
        self.assertListEqual(
            list(dataset._get_label_list()), ['tag_a', 'tag_b'])

    def test__get_num_samples_is_correct(self):
        pd.read_csv = MagicMock(return_value=pd.DataFrame({
            'post_id': ['id_1', 'id_2', 'id_3', 'idç_4', 'id_5', 'id_6', 'id_7', 'id_8', 'id_9', 'id_10'],
            'tag_name': ['tag_a', 'tag_b', 'tag_a', 'tag_b', 'tag_a', 'tag_b', 'tag_a', 'tag_b', 'tag_a', 'tag_b'],
            'tag_id': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            'tag_position': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            'title': ['title_1', 'title_2', 'title_3', 'title_4', 'title_5', 'title_6', 'title_7', 'title_8', 'title_9', 'title_10']
        }))
        dataset = utils.LocalTextCategorizationDataset(
            "fake_path", batch_size=2, min_samples_per_label=1)
        self.assertEqual(dataset._get_num_samples(), 10)

    def test_get_train_batch_returns_expected_shape(self):

        pd.read_csv = MagicMock(return_value=pd.DataFrame({
            'post_id': ['id_1', 'id_2', 'id_3', 'idç_4', 'id_5', 'id_6', 'id_7', 'id_8', 'id_9', 'id_10'],
            'tag_name': ['tag_a', 'tag_b', 'tag_a', 'tag_b', 'tag_a', 'tag_b', 'tag_a', 'tag_b', 'tag_a', 'tag_b'],
            'tag_id': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            'tag_position': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            'title': ['title_1', 'title_2', 'title_3', 'title_4', 'title_5', 'title_6', 'title_7', 'title_8', 'title_9', 'title_10']
        }))
        dataset = utils.LocalTextCategorizationDataset(
            "fake_path", 2, min_samples_per_label=1)

        # We have 10 values and we want batch size of 2. So after the train_test_split
        # we have 8 values for the train set and 2 values for the test set.
        # And the train set will be split into 6 values for the train set and 2 values for the validation set.
        # So we are supposed to have 3 batches for the train set and 1 batch for the validation set.

        first_x, first_y = dataset.get_train_batch()
        self.assertEqual(first_x.shape[0], 2)
        self.assertEqual(first_y.shape[0], 2)
        x, y = dataset.get_train_batch()
        self.assertEqual(x.shape[0], 2)
        self.assertEqual(y.shape[0], 2)
        x, y = dataset.get_train_batch()
        self.assertEqual(x.shape[0], 2)
        self.assertEqual(y.shape[0], 2)
        x, y = dataset.get_train_batch()
        self.assertEqual(x.shape[0], 2)
        self.assertEqual(y.shape[0], 2)
        x, y = dataset.get_train_batch()
        # we have 3 batches for the train set, so if we exceed that number, we should get the first batch again
        self.assertEqual(x.tolist(), first_x.tolist())
        self.assertEqual(y.tolist(), first_y.tolist())

    def test_get_test_batch_returns_expected_shape(self):
        pd.read_csv = MagicMock(return_value=pd.DataFrame({
            'post_id': ['id_1', 'id_2', 'id_3', 'idç_4', 'id_5', 'id_6', 'id_7', 'id_8', 'id_9', 'id_10'],
            'tag_name': ['tag_a', 'tag_b', 'tag_a', 'tag_b', 'tag_a', 'tag_b', 'tag_a', 'tag_b', 'tag_a', 'tag_b'],
            'tag_id': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            'tag_position': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            'title': ['title_1', 'title_2', 'title_3', 'title_4', 'title_5', 'title_6', 'title_7', 'title_8', 'title_9', 'title_10']
        }))
        dataset = utils.LocalTextCategorizationDataset(
            "fake_path", batch_size=2, min_samples_per_label=1)

        first_x, first_y = dataset.get_test_batch()
        self.assertEqual(first_x.shape[0], 2)
        self.assertEqual(first_y.shape[0], 2)
        x, y = dataset.get_test_batch()
        self.assertEqual(x.shape[0], 2)
        self.assertEqual(y.shape[0], 2)
        x, y = dataset.get_test_batch()
        self.assertEqual(x.shape[0], 2)
        self.assertEqual(y.shape[0], 2)
        x, y = dataset.get_test_batch()
        self.assertEqual(x.shape[0], 2)
        self.assertEqual(y.shape[0], 2)
        x, y = dataset.get_test_batch()
        # we have 3 batches for the test set, so if we exceed that number, we should get the first batch again
        self.assertEqual(x.tolist(), first_x.tolist())
        self.assertEqual(y.tolist(), first_y.tolist())

    def test_get_train_batch_raises_assertion_error(self):
        pd.read_csv = MagicMock(return_value=pd.DataFrame({
            'post_id': ['id_1', 'id_2'],
            'tag_name': ['tag_a', 'tag_b'],
            'tag_id': [1, 2],
            'tag_position': [0, 0],
            'title': ['title_1', 'title_2']
        }))

        with self.assertRaises(AssertionError):
            utils.LocalTextCategorizationDataset(
                "fake_path", batch_size=3, min_samples_per_label=1)


if __name__ == "__main__":
    unittest.main()

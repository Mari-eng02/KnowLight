import os
import sys

# Add root workspace directory to sys.path
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from datarec.datasets import AmazonToysAndGames
from datarec.processing.kcore import UserItemIterativeKCore
from datarec.processing.binarizer import Binarize
from datarec.splitters.user_stratified.temporal.leave_out import LeaveOneLast
from datarec.io.writers import write_transactions_tabular as write_tabular
from knowlight.paths import AMAZONTOYSANDGAMES, dataset_directory, dataset_filepath
from knowlight.datasets.metadata_loader import save_item_metadata_from_datarec

dataset_name = AMAZONTOYSANDGAMES

raw_dataset = AmazonToysAndGames(version="2023")
dataset = raw_dataset.prepare_and_load()

binarizer = Binarize(threshold=3, keep='positive')
dataset = binarizer.run(datarec=dataset)

kcore = UserItemIterativeKCore(cores=[10, 10])
dataset = kcore.run(dataset)

print('Train validation test splitting...')
spl = LeaveOneLast(test=True, validation=True, seed=42) 
split = spl.run(dataset)

print('Train validation test splitting done')
print('Writing dataset...')

dataset_dir = dataset_directory(dataset_name=dataset_name, create_if_not_exists=True)
print(f'Dataset directory: {dataset_dir}')

write_tabular(dataset,
              filepath=dataset_filepath(dataset_name, 'processed', exists=False),
              sep='\t',
              header=False,
              decimal='.',
              include_user=True,
              include_item=True,
              include_rating=True,
              include_timestamp=True)

write_tabular(split['train'].to_rawdata(),
              filepath=dataset_filepath(dataset_name, 'train', exists=False),
              sep='\t',
              header=False,
              decimal='.',
              include_user=True,
              include_item=True,
              include_rating=True,
              include_timestamp=True)

write_tabular(split['test'].to_rawdata(),
              filepath=dataset_filepath(dataset_name, 'test', exists=False),
              sep='\t',
              header=False,
              decimal='.',
              include_user=True,
              include_item=True,
              include_rating=True,
              include_timestamp=True)

write_tabular(split['val'].to_rawdata(),
              filepath=dataset_filepath(dataset_name, 'val', exists=False),
              sep='\t',
              header=False,
              decimal='.',
              include_user=True,
              include_item=True,
              include_rating=True,
              include_timestamp=True)

# Save item metadata json
save_item_metadata_from_datarec(dataset_name, raw_dataset, dataset_dir)

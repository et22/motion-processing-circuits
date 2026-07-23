from pathlib import Path

def concatenate_nidq_bin_files(bin_path1, bin_path2, output_path):
    with open(bin_path1, 'rb') as f1, open(bin_path2, 'rb') as f2, open(output_path, 'wb') as out:
        # Copy contents of first file
        while chunk := f1.read(1024 * 1024):
            out.write(chunk)
        # Copy contents of second file
        while chunk := f2.read(1024 * 1024):
            out.write(chunk)

date = '0204'
bin_path1 = f'../../Downloads/{date}25_rf_g0_t0.nidq.bin'
bin_path2 = f'../../Downloads/{date}25_g0_t0.nidq.bin'#f'./data/rawdata/sync_data/old/{date}25_g0_t0.nidq.bin'

output_path = f'./data/rawdata/sync_data/{date}25_g0_t0.nidq.bin' # this is a little dangerous, be careful! make SURE that this does not exist when you run it, otherwise you will overwrite

concatenate_nidq_bin_files(bin_path1, bin_path2, output_path)
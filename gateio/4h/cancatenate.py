import glob2

filenames = sorted(glob2.glob('4h_*.csv'))  # list of all .txt files in the directory
print(filenames)
with open('4h_BTC_USDT-202101-202402.csv', 'w') as outfile:
    outfile.write("timestamp,volume,close,high,low,open\n")
    for fname in filenames:
        with open(fname) as infile:
            for line in infile:
                outfile.write(line)
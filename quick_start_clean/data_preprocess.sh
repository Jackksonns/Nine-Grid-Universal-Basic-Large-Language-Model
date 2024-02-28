for i in {1..10};do
cat raw_data/alpaca_zh.jsonl >> raw_data/alpaca_zh_repeat.jsonl
done

mkdir raw_data_repeat
mv raw_data/alpaca_zh_repeat.jsonl raw_data_repeat/data.jsonl

python data_binarize.py --input raw_data_repeat --data_type json --output_path bin_data_repeat --output_name data

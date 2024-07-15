for i in {1..10};do
cat raw_data/alpaca_zh.jsonl >> raw_data/alpaca_zh_repeat.jsonl
done

mkdir raw_data_repeat
mv raw_data/alpaca_zh_repeat.jsonl raw_data_repeat/data.jsonl


python convert_json2index.py --path raw_data_repeat/data.jsonl --language en --output alpaca_zh_repeat

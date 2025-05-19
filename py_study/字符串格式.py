name = "风"
stock_price = 9.99
stock_code = "003025" #注意，此处应该是用字符串定义，因为如果第一位是0，用数字不合规
stock_price_daily_growth_factor = 1.2
growth_days = 7

print(f"公司：{name},股票代码：{stock_code},当前股价：{stock_price}")
print("每次增长系数是%.2f,经过%d天增长,股价达到了%.2f" %(stock_price_daily_growth_factor,growth_days,stock_price*stock_price_daily_growth_factor**growth_days))
# DF DESCRIPTION
1) AGG_TRADE
   contains all trades by timestamp, with buy / sell order details. There's two aggregated quantities each day, with side == ''
2) BBO
   contains best bid and best ask by timestamp for the selected day
3) BBO_DELTA
   contains best bid and best ask deltas by timestamp for the selected day
4) MICRO_PX
   According to ChatGPT (double check!) it is an indicator of the order book imbalance, calculated with some formula (which we dont know)
5) OFI
   order flow imbalance (delta bid volume - delta ask volume) -> double check!
6) ORDER_SUMMARY
   contains details of added / deleted orders -> TRD orders are the one executed (roughly 3k / 83k total)
7) L2_SNAPSHOT
   level 5 (lol) order book details. Apparently there are some changes in the positions 
   (either queues sizes or prices) even if the timestamp is exactly! the same. 
   The fact that bid / ask prices are equally spaced is normal: in dense markets, all levels move by the same tick. To check this:
   plt.figure(); plt.plot(df_map["L2_SNAPSHOT"]["askPx_0"].astype(float), label="0"); plt.plot(df_map["L2_SNAPSHOT"]["askPx_1"].astype(float), label="1"); plt.plot(df_map["L2_SNAPSHOT"]["askPx_2"].astype(float), label="2"); plt.plot(df_map["L2_SNAPSHOT"]["askPx_3"].astype(float), label="3"); plt.plot(df_map["L2_SNAPSHOT"]["askPx_4"].astype(float), label="4"); plt.legend(); plt.show()

In general, check for artifacts (weird values ex. price falling back to opening price, etc.) and general cleaness of the dataset
Check extractDataFramesFromVar to see how it manipulates it - also there were some timestamp matches in the var files
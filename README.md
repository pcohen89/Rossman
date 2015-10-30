# Rossman

Kaggle competition. Will fill this out later

## Ideas
- [ ] looks like we may want to do something which measures days since promo re-up
- [ ] recency weighting
- [ ] add lags which are actually usable for prediction
- [ ] ln(sales) as outcome, may
- [ ] train linear models on every store
- [ ] use customer groupings to create store groupings; run linear models on each store grouping
- [ ] NN, maybe start with: 4 features input [74 250 250 250 1] with 10% dropout in each layer, and sigmoid activation (lelu?) https://www.kaggle.com/c/rossmann-store-sales/forums/t/16928/neural-network-models
- [ ] google trends (how is this merged on?)
- [ ] https://www.kaggle.com/c/rossmann-store-sales/forums/t/17048/putting-stores-on-the-map/96627#post96627
- [ ] tag stores that are open on sundays
- [ ] day of week
- [ ] fit a first stage model in the train check its score tune it whatever, but then fit a random forest in the validation including the predictions from teh first stage model
- [ ] was store closed on previous day

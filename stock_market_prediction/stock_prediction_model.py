import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

gold_train_data = 'Stock Market Prediction Model/CSV_Files/Gold Data Last Year.csv'
gold_test_data = 'Stock Market Prediction Model/CSV_Files/Gold Data Last Month.csv'

gas_train_data = 'Stock Market Prediction Model/CSV_Files/Gas Data Last Year.csv'
gas_test_data = 'Stock Market Prediction Model/CSV_Files/Gas Data Last Month.csv'

oil_train_data = 'Stock Market Prediction Model/CSV_Files/Oil Data Last Year.csv'
oil_test_data = 'Stock Market Prediction Model/CSV_Files/Oil Data Last Month.csv'

silver_train_data = 'Stock Market Prediction Model/CSV_Files/Silver Data Last Year.csv'
silver_test_data = 'Stock Market Prediction Model/CSV_Files/Silver Data Last Month.csv'


current_train_data = gold_train_data
current_test_data = gold_test_data

num_train_data_points = 266
num_test_data_points = 22

learning_rate = 0.1
num_epochs = 100


def load_stock_data(stock_name, num_data_points):
    data = pd.read_csv(stock_name,
                       skiprows=0,
                       nrows=num_data_points,
                       usecols=['Price', 'Open', 'Vol.'])
    final_price = data['Price'].astype(str).str.replace(',', '').astype(np.float)
    opening_price = data['Open'].astype(str).str.replace(',', '').astype(np.float)
    volumes = data['Vol.'].str.strip('MK').astype(np.float)
    return final_price, opening_price, volumes


def calc_price_diff(final_prices, opening_prices):
    return [
        opening_prices[d_i + 1] - final_prices[d_i]
        for d_i in range(len(final_prices) - 1)]


def calc_accuracy(expected_values, actual_values):
    num_correct = 0
    for a_i in range(len(actual_values)):
        if ((actual_values[a_i] < 0 < expected_values[a_i]) or
                (actual_values[a_i] > 0 > expected_values[a_i])):
            num_correct += 1
    return (num_correct/len(actual_values)) * 100


train_final_prices, train_opening_prices, train_volumes = load_stock_data(
    current_train_data, num_train_data_points)
train_price_diffs = calc_price_diff(train_final_prices, train_opening_prices)
train_volumes = train_volumes[:-1]

test_final_prices, test_opening_prices, test_volumes = load_stock_data(
    current_test_data, num_test_data_points)
test_price_diffs = calc_price_diff(test_final_prices, test_opening_prices)
test_volumes = test_volumes[:-1]

x = tf.placeholder(dtype=tf.float32, name='x')
W = tf.Variable([0.1], name='W')
b = tf.Variable([0.1], name='b')
y = W * x + b
y_predicted = tf.placeholder(dtype=tf.float32, name='y_predicted')

loss = tf.reduce_sum(tf.square(y - y_predicted))
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)

session = tf.Session()
session.run(tf.global_variables_initializer())
for _ in range(num_epochs):
    session.run(optimizer, feed_dict={x: train_volumes, y_predicted: train_price_diffs})
results = session.run(y, feed_dict={x: test_volumes})
accuracy = calc_accuracy(test_price_diffs, results)
print("accuracy of model: {0:.2f}%".format(accuracy))

# plt.figure(1)
# plt.plot(train_volumes, train_price_diffs, 'bo')
# plt.title("Price Difference for Given Volume for the Past Year")
# plt.xlabel('Volumes')
# plt.ylabel('Price Difference')
# plt.show()

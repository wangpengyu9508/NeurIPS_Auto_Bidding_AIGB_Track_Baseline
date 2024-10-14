import time
import gin
import numpy as np
import os
import psutil
# from saved_model.DTtest.dt import DecisionTransformer
from bidding_train_env.baseline.dt.dt import DecisionTransformer
from bidding_train_env.strategy.base_bidding_strategy import BaseBiddingStrategy
import torch
import pickle
import math


class DtBiddingStrategy(BaseBiddingStrategy):
    """
    Decision-Transformer-PlayerStrategy
    """

    def __init__(self, budget=100, name="Decision-Transformer-PlayerStrategy", cpa=2, category=1):
        super().__init__(budget, name, cpa, category)

        file_name = os.path.dirname(os.path.realpath(__file__))
        dir_name = os.path.dirname(file_name)
        dir_name = os.path.dirname(dir_name)
        model_path = os.path.join(dir_name, "saved_model", "DTtest", "dt.pt")
        picklePath = os.path.join(dir_name, "saved_model", "DTtest", "normalize_dict.pkl")

        with open(picklePath, 'rb') as f:
            normalize_dict = pickle.load(f)
        self.model = DecisionTransformer(state_dim=16, act_dim=1, state_mean=normalize_dict["state_mean"],
                                         state_std=normalize_dict["state_std"])
        self.model.load_net(model_path)

    def reset(self):
        self.remaining_budget = self.budget

    def bidding(self, timeStepIndex, pValues, pValueSigmas, historyPValueInfo, historyBid,
                historyAuctionResult, historyImpressionResult, historyLeastWinningCost):
        """
        Bids for all the opportunities in a delivery period

        parameters:
         @timeStepIndex: the index of the current decision time step.
         @pValues: the conversion action probability.
         @pValueSigmas: the prediction probability uncertainty.
         @historyPValueInfo: the history predicted value and uncertainty for each opportunity.
         @historyBid: the advertiser's history bids for each opportunity.
         @historyAuctionResult: the history auction results for each opportunity.
         @historyImpressionResult: the history impression result for each opportunity.
         @historyLeastWinningCosts: the history least wining costs for each opportunity.

        return:
            Return the bids for all the opportunities in the delivery period.
        """
        time_left = (48 - timeStepIndex) / 48
        budget_left = self.remaining_budget / self.budget if self.budget > 0 else 0
        history_xi = [result[:, 0] for result in historyAuctionResult]
        history_pValue = [result[:, 0] for result in historyPValueInfo]
        history_conversion = [result[:, 1] for result in historyImpressionResult]

        historical_xi_mean = np.mean([np.mean(xi) for xi in history_xi]) if history_xi else 0

        historical_conversion_mean = np.mean(
            [np.mean(reward) for reward in history_conversion]) if history_conversion else 0

        historical_LeastWinningCost_mean = np.mean(
            [np.mean(price) for price in historyLeastWinningCost]) if historyLeastWinningCost else 0

        historical_pValues_mean = np.mean([np.mean(value) for value in history_pValue]) if history_pValue else 0

        historical_bid_mean = np.mean([np.mean(bid) for bid in historyBid]) if historyBid else 0

        def mean_of_last_n_elements(history, n):
            # last_three_data = history[max(0, n - 3):n]
            last_three_data = history[-n:]
            if len(last_three_data) == 0:
                return 0
            else:
                return np.mean([np.mean(data) for data in last_three_data])

        last_three_xi_mean = mean_of_last_n_elements(history_xi, 3)
        last_three_conversion_mean = mean_of_last_n_elements(history_conversion, 3)
        last_three_LeastWinningCost_mean = mean_of_last_n_elements(historyLeastWinningCost, 3)
        last_three_pValues_mean = mean_of_last_n_elements(history_pValue, 3)
        last_three_bid_mean = mean_of_last_n_elements(historyBid, 3)

        current_pValues_mean = np.mean(pValues)
        current_pv_num = len(pValues)

        historical_pv_num_total = sum(len(bids) for bids in historyBid) if historyBid else 0
        last_three_ticks = slice(max(0, timeStepIndex - 3), timeStepIndex)
        last_three_pv_num_total = sum(
            [len(historyBid[i]) for i in range(max(0, timeStepIndex - 3), timeStepIndex)]) if historyBid else 0

        test_state = np.array([
            time_left, budget_left, historical_bid_mean, last_three_bid_mean,
            historical_LeastWinningCost_mean, historical_pValues_mean, historical_conversion_mean,
            historical_xi_mean, last_three_LeastWinningCost_mean, last_three_pValues_mean,
            last_three_conversion_mean, last_three_xi_mean,
            current_pValues_mean, current_pv_num, last_three_pv_num_total,
            historical_pv_num_total
        ])

        if timeStepIndex == 0:
            self.model.init_eval()
        
        # adv = 0
        # if self.category == 0 and self.cpa == 100 and self.budget == 2900:
        #     adv = 0
        # elif self.category == 0 and self.cpa == 70 and self.budget == 4350:
        #     adv = 1
        # elif self.category == 0 and self.cpa == 90 and self.budget == 3000:
        #     adv = 2
        # elif self.category == 0 and self.cpa == 110 and self.budget == 2400:
        #     adv = 3
        # elif self.category == 0 and self.cpa == 60 and self.budget == 4800:
        #     adv = 4
        # elif self.category == 0 and self.cpa == 130 and self.budget == 2000:
        #     adv = 5
        # elif self.category == 0 and self.cpa == 120 and self.budget == 2050:
        #     adv = 6
        # elif self.category == 0 and self.cpa == 80 and self.budget == 3500:
        #     adv = 7
        # elif self.category == 1 and self.cpa == 70 and self.budget == 4600:
        #     adv = 8
        # elif self.category == 1 and self.cpa == 130 and self.budget == 2000:
        #     adv = 9
        # elif self.category == 1 and self.cpa == 100 and self.budget == 2800:
        #     adv = 10
        # elif self.category == 1 and self.cpa == 110 and self.budget == 2350:
        #     adv = 11
        # elif self.category == 1 and self.cpa == 120 and self.budget == 2050:
        #     adv = 12
        # elif self.category == 1 and self.cpa == 90 and self.budget == 2900:
        #     adv = 13
        # elif self.category == 1 and self.cpa == 60 and self.budget == 4750:
        #     adv = 14
        # elif self.category == 1 and self.cpa == 80 and self.budget == 3450:
        #     adv = 15
        # elif self.category == 2 and self.cpa == 130 and self.budget == 2000:
        #     adv = 16
        # elif self.category == 2 and self.cpa == 80 and self.budget == 3500:
        #     adv = 17
        # elif self.category == 2 and self.cpa == 110 and self.budget == 2200:
        #     adv = 18
        # elif self.category == 2 and self.cpa == 100 and self.budget == 2700:
        #     adv = 19
        # elif self.category == 2 and self.cpa == 90 and self.budget == 3100:
        #     adv = 20
        # elif self.category == 2 and self.cpa == 120 and self.budget == 2100:
        #     adv = 21
        # elif self.category == 2 and self.cpa == 60 and self.budget == 4850:
        #     adv = 22
        # elif self.category == 2 and self.cpa == 70 and self.budget == 4100:
        #     adv = 23
        # elif self.category == 3 and self.cpa == 120 and self.budget == 2000:
        #     adv = 24
        # elif self.category == 3 and self.cpa == 60 and self.budget == 4800:
        #     adv = 25
        # elif self.category == 3 and self.cpa == 90 and self.budget == 3050:
        #     adv = 26
        # elif self.category == 3 and self.cpa == 70 and self.budget == 4250:
        #     adv = 27
        # elif self.category == 3 and self.cpa == 100 and self.budget == 2850:
        #     adv = 28
        # elif self.category == 3 and self.cpa == 110 and self.budget == 2250:
        #     adv = 29
        # elif self.category == 3 and self.cpa == 130 and self.budget == 2000:
        #     adv = 30
        # elif self.category == 3 and self.cpa == 80 and self.budget == 3900:
        #     adv = 31
        # elif self.category == 4 and self.cpa == 120 and self.budget == 2000:
        #     adv = 32
        # elif self.category == 4 and self.cpa == 90 and self.budget == 3250:
        #     adv = 33
        # elif self.category == 4 and self.cpa == 70 and self.budget == 4450:
        #     adv = 34
        # elif self.category == 4 and self.cpa == 80 and self.budget == 3550:
        #     adv = 35
        # elif self.category == 4 and self.cpa == 100 and self.budget == 2700:
        #     adv = 36
        # elif self.category == 4 and self.cpa == 110 and self.budget == 2100:
        #     adv = 37
        # elif self.category == 4 and self.cpa == 60 and self.budget == 4650:
        #     adv = 38
        # elif self.category == 4 and self.cpa == 130 and self.budget == 2000:
        #     adv = 39
        # elif self.category == 5 and self.cpa == 90 and self.budget == 3400:
        #     adv = 40
        # elif self.category == 5 and self.cpa == 100 and self.budget == 2650:
        #     adv = 41
        # elif self.category == 5 and self.cpa == 110 and self.budget == 2300:
        #     adv = 42
        # elif self.category == 5 and self.cpa == 80 and self.budget == 4100:
        #     adv = 43
        # elif self.category == 5 and self.cpa == 60 and self.budget == 4800:
        #     adv = 44
        # elif self.category == 5 and self.cpa == 70 and self.budget == 4450:
        #     adv = 45
        # elif self.category == 5 and self.cpa == 130 and self.budget == 2000:
        #     adv = 46
        # elif self.category == 5 and self.cpa == 120 and self.budget == 2050:
        #     adv = 47
        # else:
        #     return 1/0
        
        # target_return = 6
        # risk = 0.1

        # if adv == 0:
        #     target_return = 3
        #     risk = 0.1
        # elif adv == 1:
        #     target_return = 6
        #     risk = 0.1
        # elif adv == 2:
        #     target_return = 7
        #     risk = 0.1
        # elif adv == 3: 
        #     target_return = 5
        #     risk = 0.3
        # elif adv == 4: 
        #     target_return = 5
        #     risk = 0.3
        # elif adv == 5:
        #     target_return = 2
        #     risk = 0.3
        # elif adv == 6:
        #     target_return = 5
        #     risk = 0.3
        # elif adv == 7:
        #     target_return = 2
        #     risk = 0.3
        # elif adv == 8:
        #     target_return = 6
        #     risk = 0.1
        # elif adv == 9:
        #     target_return = 4
        #     risk = 0.3
        # elif adv == 10:
        #     target_return = 5
        #     risk = 0.3
        # elif adv == 11:
        #     target_return = 6
        #     risk = 0.3
        # elif adv == 12:
        #     target_return = 6
        #     risk = 0.1
        # elif adv == 13:
        #     target_return = 6
        #     risk = 0.1
        # elif adv == 14:
        #     target_return = 6
        #     risk = 0.1
        # elif adv == 15:
        #     target_return = 5
        #     risk = 0.3
        # elif adv == 16:
        #     target_return = 6
        #     risk = 0.1
        # elif adv == 17:
        #     target_return = 2
        #     risk = 0.3
        # elif adv == 18:
        #     target_return = 6
        #     risk = 0.1
        # elif adv == 19:
        #     target_return = 6
        #     risk = 0.1
        # elif adv == 20:
        #     target_return = 6
        #     risk = 0.1
        # elif adv == 21:
        #     target_return = 6
        #     risk = 0.1
        # elif adv == 22:
        #     target_return = 6
        #     risk = 0.1
        # elif adv == 23:
        #     target_return = 6
        #     risk = 0.1
        # elif adv == 24:
        #     target_return = 5
        #     risk = 0.3
        # elif adv == 25: 
        #     target_return = 4
        #     risk = 0.3
        # elif adv == 26:
        #     target_return = 5
        #     risk = 0.3
        # elif adv == 27:
        #     target_return = 6
        #     risk = 0.1
        # elif adv == 28:
        #     target_return = 5
        #     risk = 0.3
        # elif adv == 29:
        #     target_return = 5
        #     risk = 0.3
        # elif adv == 30:
        #     target_return = 6
        #     risk = 0.3
        # elif adv == 31:
        #     target_return = 5
        #     risk = 0.3
        # elif adv == 32:
        #     target_return = 5
        #     risk = 0.3
        # elif adv == 33:
        #     target_return = 5
        #     risk = 0.3
        # elif adv == 34:
        #     target_return = 6
        #     risk = 0.1
        # elif adv == 35:
        #     target_return = 2
        #     risk = 0.3
        # elif adv == 36:
        #     target_return = 6
        #     risk = 0.1
        # elif adv == 37:
        #     target_return = 6
        #     risk = 0.3
        # elif adv == 38:
        #     target_return = 2
        #     risk = 0.3
        # elif adv == 39:
        #     target_return = 6
        #     risk = 0.1
        # elif adv == 40:
        #     target_return = 6
        #     risk = 0.1
        # elif adv == 41:
        #     target_return = 5
        #     risk = 0.3
        # elif adv == 42:
        #     target_return = 6
        #     risk = 0.1
        # elif adv == 43:
        #     target_return = 6
        #     risk = 0.1
        # elif adv == 44:
        #     target_return = 5
        #     risk = 0.3
        # elif adv == 45:
        #     target_return = 6
        #     risk = 0.1
        # elif adv == 46:
        #     target_return = 4
        #     risk = 0.3
        # elif adv == 47:
        #     target_return = 6
        #     risk = 0.1
        
        alpha = self.model.take_actions(test_state, target_return = 0,
                                        pre_reward=sum(history_conversion[-1]) if len(history_conversion) != 0 else None)
        
        # richiness = budget_left / time_left - 1
        # risk_tendency = math.tanh(0.1 * richiness)
        # pValues = pValues * (1 + risk_tendency)

        bids = alpha * pValues

        return bids



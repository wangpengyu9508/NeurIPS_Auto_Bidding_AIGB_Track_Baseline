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
import json
import traceback

class DtBiddingStrategy(BaseBiddingStrategy):
    """
    Decision-Transformer-PlayerStrategy
    """

    def __init__(self, budget=100, name="Decision-Transformer-PlayerStrategy", cpa=2, category=1):
        super().__init__(budget, name, cpa, category)

        file_name = os.path.dirname(os.path.realpath(__file__))
        dir_name = os.path.dirname(file_name)
        dir_name = os.path.dirname(dir_name)
        model_path = os.path.join(dir_name, "saved_model", "DTtest", "dt_dt.pt")
        picklePath = os.path.join(dir_name, "saved_model", "DTtest", "normalize_dict_dt.pkl")

        with open(picklePath, 'rb') as f:
            normalize_dict = pickle.load(f)
        self.model = DecisionTransformer(state_dim=16, act_dim=1, state_mean=normalize_dict["state_mean"],
                                         state_std=normalize_dict["state_std"])
        self.model.load_net(model_path)
        # with open(os.path.join(dir_name, "saved_model", "DTtest", "adv_period.json"), 'r') as file:
        #     self.adv_period = json.load(file)

        self.adv_period = {
            "0_100_2900": 0,
            "0_70_4350": 0,
            "0_90_3000": 0,
            "0_110_2400": 0,
            "0_60_4800": 0,
            "0_130_2000": 0,
            "0_120_2050": 0,
            "0_80_3500": 0,
            "1_70_4600": 0,
            "1_130_2000": 0,
            "1_100_2800": 0,
            "1_110_2350": 0,
            "1_120_2050": 0,
            "1_90_2900": 0,
            "1_60_4750": 0,
            "1_80_3450": 0,
            "2_130_2000": 0,
            "2_80_3500": 0,
            "2_110_2200": 0,
            "2_100_2700": 0,
            "2_90_3100": 0,
            "2_120_2100": 0,
            "2_60_4850": 0,
            "2_70_4100": 0,
            "3_120_2000": 0,
            "3_60_4800": 0,
            "3_90_3050": 0,
            "3_70_4250": 0,
            "3_100_2850": 0,
            "3_110_2250": 0,
            "3_130_2000": 0,
            "3_80_3900": 0,
            "4_120_2000": 0,
            "4_90_3250": 0,
            "4_70_4450": 0,
            "4_80_3550": 0,
            "4_100_2700": 0,
            "4_110_2100": 0,
            "4_60_4650": 0,
            "4_130_2000": 0,
            "5_90_3400": 0,
            "5_100_2650": 0,
            "5_110_2300": 0,
            "5_80_4100": 0,
            "5_60_4800": 0,
            "5_70_4450": 0,
            "5_130_2000": 0,
            "5_120_2050": 0,
            "1_2_100": 0
        }

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
        try:
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

            # kcate = str(int(float(self.category)))
            # kcpa = str(int(float(self.cpa)))
            # kbudget = str(int(float(self.budget)))
            # key = kcate + "_" + kcpa + "_" + kbudget

            if timeStepIndex == 0:
                # self.adv_period[key] += 1
                self.model.init_eval()
            
            alpha = self.model.take_actions(test_state, target_return = 0.015,
                                            pre_reward=sum(history_conversion[-1]) if len(history_conversion) != 0 else None)
            
            # richiness = budget_left / time_left - 1
            # risk_tendency = math.tanh(0.1 * richiness)
            # pValues = pValues * (1 + risk_tendency)

            bids = alpha * pValues

            return bids
        
        except Exception as e:
            traceback.print_exc()
        



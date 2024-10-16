import numpy as np
import pandas as pd
import math
import random
import logging
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bidding_train_env.strategy import PlayerBiddingStrategy
from bidding_train_env.dataloader.test_dataloader import TestDataLoader
from bidding_train_env.environment.offline_env import OfflineEnv
import pickle

np.random.seed(2024)
random.seed(2024)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(name)s] [%(filename)s(%(lineno)d)] [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

def getScore_nips(reward, cpa, cpa_constraint):
    beta = 2
    penalty = 1
    if cpa > cpa_constraint:
        coef = cpa_constraint / (cpa + 1e-10)
        penalty = pow(coef, beta)
    return penalty * reward

def cal_bid_result(bid_results, pValues, pValueSigmas, ori_bid_results): # 当前步
    aution_res = {} # 是否曝光
    impression_res = {} # 是否转化
    cost_res = {} # 扣费
    leastWinningCost = {} # 第四价
    adslots = {} # 多坑
    for i in range(48):
        aution_res[i] = []
        impression_res[i] = []
        cost_res[i] = []
        leastWinningCost[i] = []
        adslots[i] = []
    for adv in range(48):
        final_bid_results = ori_bid_results
        assert(len(final_bid_results[adv]) == len(bid_results[adv]))
        final_bid_results[adv] = bid_results[adv]
        for pv in range(len(bid_results[0])):
            bid1, bid2, bid3, bid4 = None, None, None, None
            # 计算每个pv的第1，2，3，4价
            for j in range(48):
                if bid1 is None or final_bid_results[j][pv] >= bid1[1]:
                    bid4 = bid3
                    bid3 = bid2
                    bid2 = bid1
                    bid1 = (j, final_bid_results[j][pv])
                    continue
                if bid2 is None or final_bid_results[j][pv] >= bid2[1]:
                    bid4 = bid3
                    bid3 = bid2
                    bid2 = (j, final_bid_results[j][pv])
                    continue
                if bid3 is None or final_bid_results[j][pv] >= bid3[1]:
                    bid4 = bid3
                    bid3 = (j, final_bid_results[j][pv])
                    continue
                if bid4 is None or final_bid_results[j][pv] >= bid4[1]:
                    bid4 = (j, final_bid_results[j][pv])
                    continue
            if adv == bid1[0]:
                adslots[adv].append(1.0)
                aution_res[adv].append(1.0)
                cost_res[adv].append(bid2[1])
                p1 = np.clip(np.random.normal(loc=pValues[adv][pv], scale=pValueSigmas[adv][pv]), 0.0, 1.0)
                p1_conversion = np.random.binomial(n=1, p=p1)
                impression_res[adv].append(p1_conversion)
            elif adv == bid2[0]:
                adslots[adv].append(2)
                aution_second = 1.0 if random.random() <= 0.8 else 0.0
                aution_res[adv].append(aution_second)
                cost_second = bid3[1] if aution_second else 0.0
                cost_res[adv].append(cost_second)
                p2 = np.clip(np.random.normal(loc=pValues[adv][pv], scale=pValueSigmas[adv][pv]), 0.0, 1.0)
                p2_conversion = np.random.binomial(n=1, p=p2) if aution_second else 0.0
                impression_res[adv].append(p2_conversion)
            elif adv == bid3[0]:
                adslots[adv].append(3)
                aution_three = 1 if random.random() <= 0.4832 else 0.0
                aution_res[adv].append(aution_three)
                cost_three = bid4[1] if aution_three else 0.0
                cost_res[adv].append(cost_three)
                p3 = np.clip(np.random.normal(loc=pValues[adv][pv], scale=pValueSigmas[adv][pv]), 0.0, 1.0)
                p3_conversion = np.random.binomial(n=1, p=p3) if aution_three else 0.0
                impression_res[adv].append(p3_conversion)
            else:
                adslots[adv].append(0.0)
                aution_res[adv].append(0.0)
                cost_res[adv].append(0.0)
                impression_res[adv].append(0.0)
            leastWinningCost[adv].append(bid4[1])

    return aution_res, impression_res, cost_res, leastWinningCost, adslots


def run_test(period, read_file_path, save_file_path):
    print("开始测试period", str(period))

    raw_data = None
    # 二进制文件io更快
    raw_data_path = save_file_path + 'raw_data' + str(period) + '.pickle'
    if os.path.exists(raw_data_path):
        with open(raw_data_path, 'rb') as file:
            raw_data = pickle.load(file)
    else:
        tem = pd.read_csv(read_file_path + 'period-' + str(period) + '.csv')
        with open(raw_data_path, 'wb') as file:
            pickle.dump(tem, file)
        raw_data = tem

    # 对当前period，adv的每个step的多个pv预估处理
    grouped_data = raw_data.sort_values('pvIndex').groupby(['deliveryPeriodIndex', 'advertiserNumber', 'timeStepIndex'])
    data_dict = {key: group for key, group in grouped_data}

    # agent代表每个adv的出价历史记录
    history = {}
    for i in range(48): # adv
        history[i] = {
            'historyBids': [],
            'historyAuctionResult': [],
            'historyImpressionResult': [],
            'historyLeastWinningCost': [],
            'historyPValueInfo': []
        }

    agents = {}
    for adv in range(48):
        key = (period, adv, 0)
        cpa_constraint = data_dict[key]['CPAConstraint'].iloc[0]
        adv_budget = data_dict[key]['budget'].iloc[0]
        adv_category = data_dict[key]['advertiserCategoryIndex'].iloc[0]
        agent = PlayerBiddingStrategy(cpa=cpa_constraint, budget=adv_budget, category=adv_category)
        agents[adv] = agent

    # 统计各个agent的得分情况
    columns = ['Periods', 'timeStep', 'Adv', 'budget', 'pvs', 'pValue', 'bid', 'leastWinningCost', 'Score', 'Reward', 'budget_consumer_ratio', 'Cost', 'CPA-real', 'CPA-constraint', 'cpa_exceed_rate']
    df_detail = pd.DataFrame(columns=columns)

    # 开始重新构建新数据集合
    ac_columns = ['deliveryPeriodIndex','advertiserNumber','advertiserCategoryIndex','budget','CPAConstraint','realAllCost','realAllConversion','timeStepIndex','state','action','reward','reward_continuous','done','realTimeCost','pValue','pValuesSigmas','next_state']
    ac_df = pd.DataFrame(columns=ac_columns)

    # 为计算state准备
    state_his = {}
    for adv in range(48):
        state_his[adv] = {
            'state_his_bid': [],
            'state_his_wincost': [],
            'state_his_pvalue': [],
            'state_his_reward': [],
            'state_his_xi': [],
            'state_his_pv': [],
            'state_his_budget_left': [],
            'state_his_cost': [],
            'state_his_reward_sum': []
        }

    for t in range(48): # each time step
        print("正在处理timeStep:", str(t))
        # 初始化出价结果
        ori_bid_results = {} # 原始48个agent的出价结果
        bid_results = {} # 当前48个agent的出价结果
        pValues = {}
        pValueSigmas = {}
        for adv in range(48):
            key = (period, adv, t)
            ori_bid_results[adv] = data_dict[key]['bid'].apply(np.array).tolist() 
            pValue = data_dict[key]['pValue'].apply(np.array).tolist()
            pValueSigma = data_dict[key]['pValueSigma'].apply(np.array).tolist()
            historyPValueInfo = history[adv]['historyPValueInfo']
            historyBids = history[adv]['historyBids']
            historyAuctionResult = history[adv]['historyAuctionResult']
            historyImpressionResult = history[adv]['historyImpressionResult']
            historyLeastWinningCost = history[adv]['historyLeastWinningCost']
            if agents[adv].remaining_budget < 0.1:
                bid = np.zeros(len(pValue))
            else:
                bid = agents[adv].bidding(t, pValue, pValueSigma, historyPValueInfo, historyBids, 
                                    historyAuctionResult, historyImpressionResult, historyLeastWinningCost)
            bid_results[adv] = bid
            pValues[adv] = pValue
            pValueSigmas[adv] = pValueSigma
        # 判决
        aution_res, impression_res, cost_res, leastWinningCost, adSlots = cal_bid_result(bid_results, pValues, pValueSigmas, ori_bid_results)
        for adv in range(48):
            over_cost_ratio = max((np.sum(np.array(cost_res[adv])) - agents[adv].remaining_budget) / (np.sum(np.array(cost_res[adv])) + 1e-4), 0)
            while over_cost_ratio > 0 and agents[adv].remaining_budget > 5.0:
                pv_index = np.where(np.array(aution_res[adv]) == 1)[0]
                dropped_pv_index = np.random.choice(pv_index, int(math.ceil(pv_index.shape[0] * over_cost_ratio)),
                                                    replace=False)
                bid_results[adv][dropped_pv_index] = 0
                aution_res, impression_res, cost_res, leastWinningCost, adSlots = cal_bid_result(bid_results, pValues, pValueSigmas, ori_bid_results)
                over_cost_ratio = max((np.sum(np.array(cost_res[adv])) - agents[adv].remaining_budget) / (np.sum(np.array(cost_res[adv])) + 1e-4), 0)
                print(np.sum(np.array(cost_res[adv])), agents[adv].remaining_budget)
        # 更新历史队列, 剩余预算
        for adv in range(48):
            history[adv]['historyBids'].append(bid_results[adv])
            temAuctionResult = np.array(
            [(aution_res[adv][i], aution_res[adv][i], cost_res[adv][i]) for i in range(len(aution_res[adv]))])
            history[adv]['historyAuctionResult'].append(temAuctionResult)
            temImpressionResult = np.array([(impression_res[adv][i], impression_res[adv][i]) for i in range(len(impression_res[adv]))])
            history[adv]['historyImpressionResult'].append(temImpressionResult)
            history[adv]['historyLeastWinningCost'].append(leastWinningCost[adv])
            temHistoryPValueInfo = [(pValues[adv][i], pValueSigmas[adv][i]) for i in range(len(pValues[adv]))]
            history[adv]['historyPValueInfo'].append(np.array(temHistoryPValueInfo))
            state_his[adv]['state_his_budget_left'].append(agents[adv].remaining_budget / agents[adv].budget) if agents[adv].budget > 0 else 0
            agents[adv].remaining_budget = max(agents[adv].remaining_budget - np.sum(np.array(cost_res[adv])), 0)
            w_pvs = np.sum(np.array(aution_res[adv]))
            w_reward = np.sum(np.array(impression_res[adv]))
            w_cost = np.sum(np.array(cost_res[adv]))
            w_cpa_real = w_cost / (w_reward + 1e-10)
            w_cpa_exceed_rate = min((w_cpa_real - agents[adv].cpa) / (agents[adv].cpa + 1e-10), 4999)
            w_score = getScore_nips(w_reward, w_cpa_real, agents[adv].cpa)
            w_budget_consumer_ratio = w_cost / agents[adv].budget
            w_pValue = np.mean(np.array(pValues[adv]))
            w_pValueSigma = np.mean(np.array(pValueSigmas[adv]))
            w_bid = np.mean(np.array(bid_results[adv]))
            w_lwc = np.mean(np.array(leastWinningCost[adv]))

            df_new = pd.DataFrame({'Periods': period,
                'timeStep': t,
                'Adv': adv,
                'budget': agents[adv].budget,
                'pvs': w_pvs,
                'pValue': w_pValue,
                'bid': w_bid,
                'leastWinningCost': w_lwc,
                'Score': w_score,
                'Reward': w_reward,
                'budget_consumer_ratio': w_budget_consumer_ratio,
                'Cost': w_cost,
                'CPA-real': w_cpa_real,
                'CPA-constraint': agents[adv].cpa,
                'cpa_exceed_rate': w_cpa_exceed_rate}, index=[0])
            if len(df_detail) == 0:
                df_detail = df_new
            else:
                df_detail = pd.concat([df_detail, df_new], ignore_index=True)

            ac_reward_continuous = np.sum(np.where(np.array(aution_res[adv]) == 1, pValues[adv], 0))
            ac_done = 1.0 if t == 47.0 else 0.0
            ac_reward = np.mean(np.array(impression_res[adv]))
            ac_xi = np.mean(np.where(np.array(adSlots[adv]) > 0, np.ones_like(np.array(adSlots[adv])), np.zeros_like(np.array(adSlots[adv]))))
            ac_action = np.sum(np.array(bid_results[adv])) / np.sum(np.array(pValues[adv]))

            # state
            state_his[adv]['state_his_bid'].append(w_bid)
            state_his[adv]['state_his_wincost'].append(w_lwc)
            state_his[adv]['state_his_pvalue'].append(w_pValue)
            state_his[adv]['state_his_reward'].append(ac_reward)
            state_his[adv]['state_his_xi'].append(ac_xi)
            state_his[adv]['state_his_pv'].append(len(pValues[adv]))
            state_his[adv]['state_his_cost'].append(w_cost)
            state_his[adv]['state_his_reward_sum'].append(w_reward)

            ac_df_row = pd.DataFrame({'deliveryPeriodIndex': period,
                'advertiserNumber': adv,
                'advertiserCategoryIndex': agents[adv].category,
                'budget': agents[adv].budget,
                'CPAConstraint': agents[adv].cpa,
                'realAllCost': None,
                'realAllConversion': None,
                'timeStepIndex': t,
                'state': None,
                'action': ac_action,
                'reward': w_reward,
                'reward_continuous': ac_reward_continuous,
                'done': ac_done,
                'realTimeCost': w_cost,
                'pValue': w_pValue,
                'pValuesSigmas': w_pValueSigma,
                'next_state': None}, index=[0])
            if len(ac_df) == 0:
                ac_df = ac_df_row
            else:
                ac_df = pd.concat([ac_df, ac_df_row], ignore_index=True)

    # 更新每个adv的stat
    state = {}
    for adv in range(48):
        state[adv] = []
    for adv in range(48):
        real_all_cost = np.sum(np.array(state_his[adv]['state_his_cost']))
        real_all_conversion = np.sum(np.array(state_his[adv]['state_his_reward_sum']))
        ac_df.loc[ac_df['advertiserNumber'] == adv, 'realAllCost'] = real_all_cost
        ac_df.loc[ac_df['advertiserNumber'] == adv, 'realAllConversion'] = real_all_conversion
        for step in range(48):
            time_left = (48 - step) / 48
            budget_left = state_his[adv]['state_his_budget_left'][step]
            historical_bid_mean = np.mean(np.array(state_his[adv]['state_his_bid'][:step])) if len(state_his[adv]['state_his_bid'][:step]) > 0 else 0
            last_three_bid_mean = np.mean(np.array(state_his[adv]['state_his_bid'][max(step-3,0):step])) if len(state_his[adv]['state_his_bid'][max(step-3,0):step]) > 0 else 0
            historical_LeastWinningCost_mean = np.mean(np.array(state_his[adv]['state_his_wincost'][:step])) if len(state_his[adv]['state_his_wincost'][:step]) > 0 else 0
            historical_pValues_mean = np.mean(np.array(state_his[adv]['state_his_pvalue'][:step])) if len(state_his[adv]['state_his_pvalue'][:step]) > 0 else 0
            historical_conversion_mean = np.mean(np.array(state_his[adv]['state_his_reward'][:step])) if len(state_his[adv]['state_his_reward'][:step]) > 0 else 0
            historical_xi_mean = np.mean(np.array(state_his[adv]['state_his_xi'][:step])) if len(state_his[adv]['state_his_xi'][:step]) > 0 else 0
            last_three_LeastWinningCost_mean = np.mean(np.array(state_his[adv]['state_his_wincost'][max(step-3,0):step])) if len(state_his[adv]['state_his_wincost'][max(step-3,0):step]) > 0 else 0
            last_three_pValues_mean = np.mean(np.array(state_his[adv]['state_his_pvalue'][max(step-3,0):step])) if len(state_his[adv]['state_his_pvalue'][max(step-3,0):step]) > 0 else 0
            last_three_conversion_mean = np.mean(np.array(state_his[adv]['state_his_reward'][max(step-3,0):step])) if len(state_his[adv]['state_his_reward'][max(step-3,0):step]) > 0 else 0
            last_three_xi_mean = np.mean(np.array(state_his[adv]['state_his_xi'][max(step-3,0):step])) if len(state_his[adv]['state_his_xi'][max(step-3,0):step]) > 0 else 0
            current_pValues_mean = state_his[adv]['state_his_pvalue'][step]
            current_pv_num = state_his[adv]['state_his_pv'][step]
            last_three_pv_num_total = np.sum(np.array(state_his[adv]['state_his_pv'][max(step-3,0):step]))
            historical_pv_num_total = np.sum(np.array(state_his[adv]['state_his_pv'][:step]))
            state[adv].append((time_left, budget_left, historical_bid_mean, last_three_bid_mean, historical_LeastWinningCost_mean, 
                               historical_pValues_mean, historical_conversion_mean, historical_xi_mean, last_three_LeastWinningCost_mean, 
                               last_three_pValues_mean,last_three_conversion_mean, last_three_xi_mean, current_pValues_mean, 
                               current_pv_num, last_three_pv_num_total, historical_pv_num_total))
    for adv in range(48):
        for step in range(48):
            ac_df.loc[(ac_df['advertiserNumber'] == adv) & (ac_df['timeStepIndex'] == step), 'state'] = str(state[adv][step])
            if step == 47:
                ac_df.loc[(ac_df['advertiserNumber'] == adv) & (ac_df['timeStepIndex'] == step), 'next_state'] = None
            else:
                ac_df.loc[(ac_df['advertiserNumber'] == adv) & (ac_df['timeStepIndex'] == step), 'next_state'] = str(state[adv][step+1])
            

    result = df_detail.groupby(['Adv']).sum().reset_index()
    result['Periods'] = result['Periods'] / 48
    result['pValue'] = result['pValue'] / 48
    result['bid'] = result['bid'] / 48
    result['leastWinningCost'] = result['leastWinningCost'] / 48
    result['timeStep'] = '0-47'
    result['budget'] = result['budget'] / 48
    result['budget_consumer_ratio'] = result['Cost'] / result['budget']
    result['CPA-real'] = result['Cost'] / (result['Reward'] + 1e-10)
    result['CPA-constraint'] = result['CPA-constraint'] / 48
    result['cpa_exceed_rate'] = (result['CPA-real'] - result['CPA-constraint']) / (result['CPA-constraint'] + 1e-10)
    result['Score'] = result['Reward']
    coef = result['CPA-constraint'] / (result['CPA-real'] + 1e-10)
    penalty =  coef * coef * result['Reward']
    result['Score'] = result['Reward'].where(result['CPA-real'] < result['CPA-constraint'], penalty)
    result.to_csv(save_file_path + 'testPeriodAll' + str(period) + '.csv', index=False, encoding='utf-8')
    mean_score = result['Score'].mean()

    ac_df_sorted = ac_df.sort_values(by=['advertiserNumber','timeStepIndex'])
    ac_df_sorted.to_csv(save_file_path + 'train/period' + str(period) + '-rlData.csv', index=False, encoding='utf-8')

    print("test done, mean_score: ", mean_score)


if __name__ == '__main__':
    
    root_dir = "/Users/wangpengyu03/NeurIPS_Auto_Bidding_Aigb_Track_Baseline/data/traffic/"
    
    for i in range(6): # 一共21个period
        period = i + 7
        run_test(period, read_file_path = root_dir, save_file_path = root_dir + '/auction_data/')
    
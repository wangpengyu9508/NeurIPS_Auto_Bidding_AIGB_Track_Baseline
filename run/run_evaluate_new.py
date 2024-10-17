import numpy as np
import pandas as pd
import math
import logging
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bidding_train_env.strategy import PlayerBiddingStrategy
from bidding_train_env.dataloader.test_dataloader import TestDataLoader
from bidding_train_env.environment.offline_env import OfflineEnv

np.random.seed(1)

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


def run_test(data_loader, env, i, df, adv_index = 0): # 测试每一个adv-period
    """
    offline evaluation
    """
    period_adv = data_loader.keys[adv_index] # 每个period，每个adv的48步，每一步包含很多pv
    cpa_constraint = data_loader.test_dict[period_adv]['CPAConstraint'].iloc[0] # 每一步所有pv的cpa限制，对于同一个adv是一样的
    adv_budget = data_loader.test_dict[period_adv]['budget'].iloc[0] # 每一步所有pv的预算，对于同一个adv是一样的
    adv_category = data_loader.test_dict[period_adv]['advertiserCategoryIndex'].iloc[0] # 每一步所有pv对应类别，对于同一个adv是一样的
    agent = PlayerBiddingStrategy(cpa=cpa_constraint, budget=adv_budget, category=adv_category) # 一个类别，cpa限制，预算一一对应一个adv
    num_timeStepIndex, pValues, pValueSigmas, leastWinningCosts = data_loader.mock_data(period_adv) # 48，48步&每一步所有pv的pValue
    # 每一步，48个agent对所有的pv都得出一个价
    # for each adv-period, pValues = [[],[],..,[]] len = step(48), pValues[0] len = step0 pv_num
    rewards = np.zeros(num_timeStepIndex) # for each adv-period，记录每一步的rewards
    history = {
        'historyBids': [],
        'historyAuctionResult': [],
        'historyImpressionResult': [],
        'historyLeastWinningCost': [],
        'historyPValueInfo': [],
    }
    for timeStep_index in range(num_timeStepIndex): # 开始循环每一步，每一步都是多个pv
        pValue = pValues[timeStep_index]
        pValueSigma = pValueSigmas[timeStep_index]
        leastWinningCost = leastWinningCosts[timeStep_index]
        if agent.remaining_budget < env.min_remaining_budget: # 预算花完，不出价了
            bid = np.zeros(pValue.shape[0])
        else:
            bid = agent.bidding(timeStep_index, pValue, pValueSigma, history["historyPValueInfo"],
                                history["historyBids"],
                                history["historyAuctionResult"], history["historyImpressionResult"],
                                history["historyLeastWinningCost"])
        tick_value, tick_cost, tick_status, tick_conversion = env.simulate_ad_bidding(pValue, pValueSigma, bid,
                                                                                      leastWinningCost)
        over_cost_ratio = max((np.sum(tick_cost) - agent.remaining_budget) / (np.sum(tick_cost) + 1e-4), 0)
        # 当前如果超支，随机退换已竞胜的pv，直达不超过剩余预算
        while over_cost_ratio > 0:
            pv_index = np.where(tick_status == 1)[0]
            dropped_pv_index = np.random.choice(pv_index, int(math.ceil(pv_index.shape[0] * over_cost_ratio)),
                                                replace=False)
            bid[dropped_pv_index] = 0
            tick_value, tick_cost, tick_status, tick_conversion = env.simulate_ad_bidding(pValue, pValueSigma, bid,
                                                                                          leastWinningCost)
            over_cost_ratio = max((np.sum(tick_cost) - agent.remaining_budget) / (np.sum(tick_cost) + 1e-4), 0)
        agent.remaining_budget -= np.sum(tick_cost) # 扣除该步产生的费用
        rewards[timeStep_index] = np.sum(tick_conversion) # 累计该步实际转化的个数
        temHistoryPValueInfo = [(pValue[i], pValueSigma[i]) for i in range(pValue.shape[0])]
        history["historyPValueInfo"].append(np.array(temHistoryPValueInfo)) # 当前步所有pv对应的pValue list放到历史
        history["historyBids"].append(bid)
        history["historyLeastWinningCost"].append(leastWinningCost)
        temAuctionResult = np.array(
            [(tick_status[i], tick_status[i], tick_cost[i]) for i in range(tick_status.shape[0])])
        history["historyAuctionResult"].append(temAuctionResult)
        temImpressionResult = np.array([(tick_conversion[i], tick_conversion[i]) for i in range(pValue.shape[0])])
        history["historyImpressionResult"].append(temImpressionResult)
    all_reward = np.sum(rewards)
    all_cost = agent.budget - agent.remaining_budget
    budget_consumer_ratio = all_cost / agent.budget
    cpa_real = all_cost / (all_reward + 1e-10)
    cpa_constraint = agent.cpa
    cpa_exceed_rate = (cpa_real - cpa_constraint) / (cpa_constraint + 1e-10)
    score = getScore_nips(all_reward, cpa_real, cpa_constraint)
    bid_mean = np.mean([np.mean(b) for b in history['historyBids']])
    leastwincost = np.mean([np.mean(b) for b in history['historyLeastWinningCost']])
    df_new = pd.DataFrame({'epoch': i,
        'Periods': period_adv[0],
        'Adv': period_adv[1],
        'Score':score,
        'Reward': all_reward,
        'cpa_exceed_rate': cpa_exceed_rate,
        'budget_consumer_ratio': budget_consumer_ratio,
        'Cost': all_cost,
        'CPA-real': cpa_real,
        'CPA-constraint': cpa_constraint,
        'bid': bid_mean,
        'leastwincost': leastwincost}, index=[0])
    if len(df) == 0:
        df = df_new
    else:
        df = pd.concat([df, df_new], ignore_index=True)
    return df

def run_mult_adv(file_path, test_epoch, save_path=None): # 循环测试每一个adv-period
    print(file_path.split('/')[-1])
    data_loader = TestDataLoader(file_path)
    env = OfflineEnv()
    columns = ['epoch','Periods', 'Adv', 'Score', 'Reward', 'cpa_exceed_rate', 'budget_consumer_ratio', 'Cost', 'CPA-real', 'CPA-constraint']
    df = pd.DataFrame(columns=columns)
    for adv_index in range(len(data_loader.keys)): # keys = 'deliveryPeriodIndex', 'advertiserNumber'
        for i in range(test_epoch):
            df = run_test(data_loader, env, i, df, adv_index)
    result = df.groupby(['Adv']).mean().reset_index()
    result2 = result.groupby(['Periods']).mean().reset_index()
    score_all = result2['Score'].sum()
    if save_path:
        result.to_csv(save_path, index=False, encoding='utf-8')
    return score_all


if __name__ == '__main__':
    
    root_dir = "/Users/wangpengyu03/NeurIPS_Auto_Bidding_AIGB_Track_Baseline/data/traffic/"
    # data_list = ['period-7.csv', 'period-8.csv', 'period-9.csv', 'period-10.csv', 'period-11.csv', 'period-12.csv', 'period-13.csv']
    data_list = ['period-13.csv']
    score = []

    for i in range(len(data_list)):
        cur_score = run_mult_adv(file_path = root_dir + data_list[i], test_epoch=1, save_path = root_dir + "analysis-"+ data_list[i] + ".csv")
        print(cur_score)
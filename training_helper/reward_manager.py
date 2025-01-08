import numpy as np
from constants import constants as c


class CalculateReward:
    def __init__(self, max_grad, scal_fac, grad, mid):
        self.max_grad = max_grad
        self.scal_fac = scal_fac
        self.grad = grad
        self.mid = mid

        self.const = None
        self.split_const = None
        self.instr_weight = None
        self.curr_instr = None
        self.curr_instr_score = None

    def calc_reward(self, curr_row, next_row, action_idx, row_idx, instr_weight, curr_instr, eps):
        """
        Calculates the reward for selecting correct or wrong decisions.

        :param curr_row: Contains initial content
        :type curr_row: Dataframe

        :param next_row: Contains next row's content
        :type next_row: Dataframe

        :param action_idx: Index of constant.AVAILABLE_ACTIONS
        :type action_idx: int

        :param instr_weight: List of instr weight
        :type instr_weight: List[float]

        :param curr_instr: Instrument selected for current row
        :type curr_instr: int

        :return: Return positive or negative-profit
        :rtype: float

        :param eps: Contains the number of which eps the training is currently at.
        :type eps: Int

        :param row_idx: Contains the idx of the dataframe at which the training is currently at.
        :type row_idx: Int

        :return: instr_weight: Contains a list of weights allocated to instr
        :rtype: instr_weight: list[float]
        """

        current_price = curr_row['Mid Price']
        next_price = next_row['Mid Price']

        # Rewards/Punishes the instr.
        buy_and_incr = action_idx == 0 and current_price < next_price
        sell_and_decr = action_idx == 1 and current_price > next_price

        if buy_and_incr or sell_and_decr:
            outcome = 1
        else:
            outcome = 0
        instr_weight = self.adjust_reward(instr_weight, curr_instr, outcome, eps, row_idx)

        # Adjusts rewards based on action and result.
        if action_idx == 0:
            reward = current_price - next_price
        elif action_idx == 1:
            reward = next_price - current_price
        else:
            reward = 0

        return reward, instr_weight

    def dynamic_alpha(self, eps, row_idx):
        """
        Calculates the dynamic alpha based on eps and call.
        Uses the sigmoid growth.
        
        :param eps: Contains the number of which eps the training is currently at.
        :type eps: Int

        :param row_idx: Contains the idx of the dataframe at which the training is currently at.
        :type row_idx: Int

        :return: Returns dynamic alpha for calculating reward/punishment
        """
        
        sigmoid_compn = self.scal_fac / (1 + np.exp(-self.grad * (eps - self.mid)))
        log_compn = np.log(row_idx + 1)
        constant = min(sigmoid_compn * log_compn, self.max_grad)
        return constant

    def adjust_reward(self, instr_weight, curr_instr, outcome, eps, row_idx):
        """
        Adjusts the table (instr_weight) accordingly depending on whether it was a right (outcome: 1) or wrong
        decision (outcome:0).
        
        :param instr_weight: List containing all the weights allocated to instrument
        :type instr_weight: list[float]

        :param curr_instr: Currently selected instrument
        :type curr_instr: int

        :param outcome: Outcome of action, right (outcome: 1) and wrong (outcome: 0)
        :type outcome: int

        :param eps: Contains the number of which eps the training is currently at.
        :type eps: Int

        :param row_idx: Contains the idx of the dataframe at which the training is currently at.
        :type row_idx: Int

        :return: instr_weight: List containing all the weights allocated to instrument
        :rtype: list[float]
        """

        self.instr_weight = instr_weight
        self.curr_instr = curr_instr

        self.curr_instr_score = self.get_curr_instr_score

        self.const = self.dynamic_alpha(eps, row_idx)

        self.split_const = self.split_weight

        if outcome == 0:  # Chose wrong action
            self.reward_other_instr()
            self.punish_main_instr()
        else:
            self.reward_main_instr()
            self.punish_other_instr()
        return instr_weight

    @property
    def split_weight(self):
        """
        Returns the weight split among all but one instr.
        
        :return: Split weight
        :rtype: float
        """
        return self.const / (len(c.AVAIL_INSTR) - 1)

    @property
    def get_curr_instr_score(self):
        return self.instr_weight[self.curr_instr]

    def punish_main_instr(self):
        """
        Punishes the main instrument for selecting a wrong outcome.
        
        :return: None 
        """
        self.instr_weight[self.curr_instr] = (self.curr_instr_score - self.const)

    def reward_other_instr(self):
        """
        When punishing the main instrument, the rest will be rewarded.
        This is to keep a fair balance such that instruments that aren't affected have a higher chance of being
        selected the next time. Furthermore, if instruments are constantly punished, and
        only rewarded when correct. They tend to reach zero quickly, as the initial phase
        is largely used to make assumptions and learning

        :return: None
        """
        remaining_instr = [
            idx for idx in range(len(c.AVAIL_INSTR))
            if idx != self.curr_instr
        ]
        for idx in remaining_instr:
            self.instr_weight[idx] += self.split_const

    def reward_main_instr(self):
        """
        Reward the main instrument for selecting a correct outcome.

        :return: None
        """
        self.instr_weight[self.curr_instr] = (self.curr_instr_score + self.const)

    def punish_other_instr(self):
        remaining_instr = [
            idx for idx in range(len(c.AVAIL_INSTR))
            if idx != self.curr_instr
        ]
        for idx in remaining_instr:
            self.instr_weight[idx] -= self.split_const

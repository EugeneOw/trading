class CalculateReward:
    @staticmethod
    def calculate_reward(current_row_content, next_row_content, current_action):
        """
        Calculates the reward for selecting correct or wrong decisions.
        But has a default penalty for holding.

        :param current_row_content: Contains initial content
        :type current_row_content: Dataframe

        :param next_row_content: Contains next row's content
        :type next_row_content: Dataframe

        :param current_action: Buy, sell or hold (Penalty 0.1)
        :type current_action: string

        :return: return positive or negative profit
        :rtype: float
        """
        current_price = current_row_content['Mid Price']
        next_price = next_row_content['Mid Price']

        if current_price is None or next_price is None:
            raise KeyError("Missing 'Mid Price' in on of the rows.")

        if not isinstance(current_action, str):
            raise TypeError(f"'current_action' should be a string, got {type(current_action)}")

        if current_action == "Buy":
            return next_price - current_price
        elif current_action == "Sell":
            return current_price - next_price
        else:
            return -0.1



class State:
    def __init__(self, instance, must_visit, cap_allow, last_visited, cur_time, cur_load, tour):
        """
        Build a State
        Note that the set of valid actions correspond to the must_visit part of the state
        :param instance: the problem instance considered
        :param must_visit: cities that still have to be visited.
        :param last_visited: the current location
        :param cur_time: the current time
        :param tour: the tour that is currently done
        """

        self.instance = instance
        self.must_visit = must_visit
        self.cap_allow = cap_allow
        self.last_visited = last_visited
        self.cur_time = cur_time
        self.tour = tour
        self.cur_load = cur_load

    def step(self, action):
        """
        Performs the transition function of the DP model
        :param action: the action selected
        :return: the new state wrt the transition function on the current state T(s,a) = s'
        """
        if (action) >= 1:
            customer = action - 1

            new_must_visit = self.must_visit - set([customer])
            new_cap_allow = self.cap_allow - set([customer])
            new_last_visited = customer
            new_cur_time = max(self.cur_time + self.instance.travel_time[self.last_visited][customer],
                           self.instance.time_windows[customer][0])
            new_tour = self.tour + [new_last_visited]

            if (customer == 0):
                new_cur_load = self.instance.capacity
            else:
                new_cur_load = self.cur_load - self.instance.demands[customer]


            #  Application of the validity conditions and the pruning rules before creating the new state
            new_cap_allow = self.prune_invalid_actions(new_must_visit, new_cap_allow, new_last_visited, new_cur_time, new_cur_load)
            # new_must_visit = self.prune_dominated_actions(new_must_visit, new_cur_time)

            new_state = State(self.instance, new_must_visit, new_cap_allow, new_last_visited, new_cur_time, new_cur_load, new_tour)
        else:
            new_state = State(self.instance, self.must_visit, self.cap_allow, self.last_visited, self.cur_time, self.cur_load, self.tour)

        return new_state

    def is_done(self, count):
        """
        :return: True iff there is no remaining actions
        
        """
        if count == 22: #TODO to dynamic
            return True
        return len(self.must_visit) == 0

    def is_success(self):
        """
        :return: True iff there is the tour is fully completed
        """

        return len(self.tour) == self.instance.n_city

    def prune_invalid_actions(self, new_must_visit, new_cap_allow, new_last_visited, new_cur_time, new_cur_load):
        """
        Validity condition: Keep only the cities that can fit in the time windows according to the travel time.
        :param new_must_visit: the cities that we still have to visit
        :param new_last_visited: the city where we are
        :param new_cur_time: the current time
        :return:
        """

        pruned_must_visit = [a for a in new_must_visit if
                             new_cur_time + self.instance.travel_time[new_last_visited][a] <= self.instance.time_windows[a][1]]

        pruned_capacity = [a for a in new_must_visit if new_cur_load > self.instance.demands[a]]

        return set(pruned_capacity)

    def prune_dominated_actions(self, new_must_visit, new_cur_time):
        """
        Pruning dominated actions: We remove all the cities having their time windows exceeded
        :param new_must_visit: the cities that we still have to visit
        :param new_cur_time: the current time
        :return:
        """

        pruned_must_visit = [a for a in new_must_visit if self.instance.time_windows[a][1] >= new_cur_time]

        return set(pruned_must_visit)

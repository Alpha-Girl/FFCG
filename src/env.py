# The cutting stock object is the environment
from gurobipy import GRB
import gurobipy as gp
import numpy as np
import Parameters


class CuttingStock(object):
    def __init__(self, customer_count, customer_length,
                 roll_length, name_):
        # each instance corresponds to self.state = self.env.reset() in learning_method in agent.py
        # state: curent graph (connection, current column nodes, current constraint node and their features)
        # static info: problem defination, same info used for initialization this instance
        self.name = name_
        self.n = len(customer_count)
        self.m = sum(customer_count)
        self.order_lens = customer_length
        self.demands = customer_count
        self.roll_len = roll_length
        # dynamic info (info needed to for state + reward), get from CG iterations from solving current RMP and PP:
        self.objVal_history = []
        self.objVal_history_int = []
        self.node_count_history = []
        self.total_steps = 0

        # action with their reduced cost (stored as tuple) ([all the patterns],[the reduced cost for all those patterns])
        self.available_action = ()
        patterns = []
        for i in range(self.n):
            pattern_ = list(np.zeros(self.n).astype(int))
            pattern_[i] = int(self.roll_len/self.order_lens[i])
            patterns.append(pattern_)
        self.current_patterns = patterns
        self.count_convergence = 0
        '''  
        Info for column and constraint node features, stored using list,length will change
            column 
                      number of constraint participation
                      current solution value (if not in the basis, 0 -> int or not)
                      columnIsNew

                      column incompatibility degree --> check this

             constraint : shadow price
                          number of columns contributing to the constraint
        '''
        # for all the columns (size change)
        self.RC = []
        self.In_Cons_Num = []
        self.ColumnSol_Val = []
        self.ColumnIs_Basic = []
        self.Waste = []
        # for all the variable that are in the basis, count the number of times it's in basis, otherwise 0
        self.stay_in = []
        self.stay_out = []
        # 1-> just left the basis in last iteration, 0 not just left
        self.just_left = []
        self.just_enter = []
        # 1-> is action node, 0 -> .. useless as we can do this at get_aug_state
        # self.action_node = []
        self.useful_pattern = []
        # for all the constraints (size fixed)
        self.Shadow_Price = []
        self.In_Cols_Num = []
        # constants
        self.pool_size = 10
        self.alpha_obj_weight = Parameters.alpha_obj_weight
        self.beta = Parameters.beta
        # save the model
        self.mainModel = None
        self.subModel = None

    def summarize(self):
        print("Problem instance with ", self.n,
              " orders and those orders include in total ", self.m, "rolls")
        print("-"*47)
        print("\nOrders:\n")
        for i, order_len in enumerate(self.order_lens):
            print("\tOrder ", i, ": length= ",
                  order_len, " demand=", self.demands[i])
        print("\nRoll Length: ", self.roll_len)

    def generate_initial_patterns(self):
        patterns = []
        for i in range(self.n):
            pattern_ = list(np.zeros(self.n).astype(int))
            pattern_[i] = int(self.roll_len/self.order_lens[i])
            patterns.append(pattern_)
        return patterns

    # get the constraint participation for each col node and col participation for each cons node
    # use current patterns to count the non-zeros in the pattern matrix

    # be careful about pointer thing (need to do the copy, otherwise previous stored value may change due to the update)
    def update_col_con_number(self, patterns):
        pa = np.asarray(patterns)
        self.In_Cons_Num = np.count_nonzero(pa, axis=1)
        self.In_Cols_Num = np.count_nonzero(pa, axis=0)

    def define_master_problem(self, patterns):

        n_pattern = len(patterns)
        pattern_range = range(n_pattern)
        order_range = range(self.n)
        patterns = np.array(patterns, dtype=int)
        master_problem = gp.Model("master problem")
        # decision variables
        lambda_ = master_problem.addVars(pattern_range,
                                         vtype=GRB.CONTINUOUS,
                                         obj=np.ones(n_pattern),
                                         name="lambda")
        # direction of optimization (min or max)
        master_problem.modelSense = GRB.MINIMIZE
        # demand satisfaction constraint
        for i in order_range:
            master_problem.addConstr(sum(patterns[p, i]*lambda_[p] for p in pattern_range) == self.demands[i],
                                     "Demand[%d]" % i)
        # master_problem.Params.LogToConsole = 1
        master_problem.Params.LogToConsole = 0
        return master_problem

    def solve_master_problem(self, patterns):
        n_pattern = len(patterns)
        pattern_range = range(n_pattern)
        order_range = range(self.n)
        patterns = np.array(patterns, dtype=int)
        master_problem = gp.Model("master problem int")
        # decision variables
        lambda_ = master_problem.addVars(pattern_range,
                                         vtype=GRB.INTEGER,
                                         obj=np.ones(n_pattern),
                                         name="lambda")

        # direction of optimization (min or max)
        master_problem.modelSense = GRB.MINIMIZE
        # demand satisfaction constraint
        for i in order_range:
            master_problem.addConstr(sum(patterns[p, i]*lambda_[p] for p in pattern_range) >= self.demands[i],
                                     "Demand[%d]" % i)
        master_problem.Params.LogToConsole = 0
        master_problem.optimize()
        list_pattern = []
        for v in master_problem.getVars():
            if v.x > 0.5:
                list_pattern.append(1)
            else:
                list_pattern.append(0)
        return master_problem.objVal, list_pattern

    def solve_subproblem_return_actions(self, duals):
        order_range = range(self.n)
        subproblem = gp.Model("subproblem")
        # Limit how many solutions to collect
        subproblem.setParam(GRB.Param.PoolSolutions, self.pool_size)
        subproblem.Params.LogToConsole = 0
        # Limit the search space by setting a gap for the worst possible solution
        # that will be accepted
        subproblem.setParam(GRB.Param.PoolSearchMode, 2)
        # decision variables
        x = subproblem.addVars(order_range,
                               vtype=GRB.INTEGER,
                               obj=duals,
                               name="x")
        # direction of optimization (min or max)
        subproblem.modelSense = GRB.MAXIMIZE
        # Length constraint
        subproblem.addConstr(
            sum(self.order_lens[i] * x[i] for i in order_range) <= self.roll_len)
        subproblem.optimize()
        nSolutions = subproblem.SolCount
        columns_to_select = []
        reduced_costs = []

        if (nSolutions >= 1):
            for i in range(nSolutions):
                sol = []
                subproblem.setParam(GRB.Param.SolutionNumber, i)
                for e in order_range:
                    sol.append(int(x[e].Xn))
                rc = 1-sum(duals[i] * sol[i] for i in range(len(duals)))
                if abs(rc - 0) <= 10**-4:
                    continue
                if rc >= 0:
                    continue
                columns_to_select.append(sol)
                reduced_costs.append(rc)
        return subproblem, columns_to_select, reduced_costs

    def basic_or_not(self):
        # use the current solution value for each column, return whether it'sin basis or not
        sol = np.asarray(self.ColumnSol_Val)
        is_basic = abs(sol - 0) >= 0.001
        integer_map = map(int, is_basic)
        integer_list = list(integer_map)
        return np.asarray(integer_list)

    def initialize(self, opt):
        self.opt = opt
        self.total_steps = 0
        # this is for taking the first step without having any actions (solving first CG iteration)
        patterns = self.current_patterns

        for i in range(len(patterns)):
            self.Waste.append(
                self.roll_len-np.dot(np.asarray(patterns[i]), np.asarray(self.order_lens)))
        self.update_col_con_number(patterns)
        # update how many constraints each columnn is in; how many columns each constraint contains
        self.mainModel = gp.Model("master problem")
        lambda_ = self.mainModel.addVars(range(len(patterns)),
                                         vtype=GRB.CONTINUOUS,
                                         obj=np.ones(len(patterns)),
                                         name="lambda")
        self.mainModel.modelSense = GRB.MINIMIZE
        for i in range(self.n):
            self.mainModel.addConstr(sum(patterns[p][i]*lambda_[p] for p in range(len(patterns))) == self.demands[i],
                                     "Demand[%d]" % i)
        self.mainModel.Params.LogToConsole = 0
        self.mainModel.optimize()
        self.ColumnSol_Val = np.asarray(self.mainModel.x)
        self.ColumnIs_Basic = np.asarray(
            self.mainModel.vbasis)+np.ones(len(patterns))
        self.objVal_history.append(self.mainModel.objVal)
        self.node_count_history.append(self.mainModel.iterCount)
        dual_variables = np.array(
            [constraint.pi for constraint in self.mainModel.getConstrs()])
        # get the rc for all the initial patterns
        for pattern in patterns:
            rc = 1-sum(dual_variables[i] * pattern[i]
                       for i in range(len(dual_variables)))
            self.RC.append(rc)
        self.Shadow_Price = dual_variables
        self.subModel = gp.Model("subproblem")
        self.subModel.setParam(GRB.Param.PoolSolutions, 10)
        self.subModel.Params.LogToConsole = 0
        self.subModel.setParam(GRB.Param.PoolSearchMode, 2)
        self.c = self.subModel.addVars(range(self.n),
                                       vtype=GRB.INTEGER,
                                       obj=dual_variables,
                                       name="x")
        self.subModel.modelSense = GRB.MAXIMIZE
        self.subModel.addConstr(
            sum(self.order_lens[i] * self.c[i] for i in range(self.n)) <= self.roll_len)
        self.subModel.optimize()
        nSolutions = self.subModel.SolCount
        columns_to_select = []
        reduced_costs = []
        if (nSolutions >= 1):
            for i in range(nSolutions):
                sol = []
                self.subModel.setParam(GRB.Param.SolutionNumber, i)
                for e in range(self.n):
                    sol.append(int(self.c[e].Xn))
                rc = 1-sum(dual_variables[i] * sol[i]
                           for i in range(len(dual_variables)))
                if abs(rc - 0) <= 10**-4:
                    continue
                if rc >= 0:
                    continue
                columns_to_select.append(sol)
                reduced_costs.append(rc)
        reward = self.subModel.Runtime + self.mainModel.Runtime
        self.available_action = (columns_to_select, reduced_costs)
        self.stay_in = list(np.zeros(len(patterns)))
        self.stay_out = list(np.zeros(len(patterns)))
        self.just_left = list(np.zeros(len(patterns)))
        self.just_enter = list(np.zeros(len(patterns)))
        is_done = False
        return reward, is_done

    def step(self, action, Train=True):
        # use this info to get stay info and just left info
        last_basis = self.ColumnIs_Basic[:]
        last_basis = np.append(last_basis, np.zeros(len(action)))
        self.total_steps += 1
        is_done = False
        for i in range(len(action)):
            self.current_patterns.append(action[i])
        patterns = self.current_patterns
        # just append one waste
        for i in range(len(action)):
            self.Waste.append(
                self.roll_len-np.dot(np.asarray(action[i]), np.asarray(self.order_lens)))

        self.update_col_con_number(patterns)
        # update how many constraints each columnn is in; how many columns each constraint contains
        for p in action:
            column = gp.Column(p, self.mainModel.getConstrs())
            self.mainModel.addVar(obj=1, vtype=GRB.CONTINUOUS, column=column)
        self.mainModel.optimize()
        self.ColumnSol_Val = np.asarray(self.mainModel.x)
        self.ColumnIs_Basic = np.asarray(
            self.mainModel.vbasis)+np.ones(len(patterns))
        # you can either stay in the basis, leave the basis, or enter the basis
        difference = last_basis - self.ColumnIs_Basic
        len_actions = len(action)
        # update the dynamic basis info based on difference
        self.just_left = list(np.zeros(len(difference)-len_actions))
        self.just_enter = list(np.zeros(len(difference)-len_actions))
        for i in range(len(difference)-len_actions):
            if difference[i] == 1:
                self.just_left[i] = 1
                self.stay_in[i] = 0
            elif difference[i] == -1:
                self.just_enter[i] = 1
                self.stay_out[i] = 0
            elif difference[i] == 0:
                if last_basis[i] == 1:
                    self.stay_in[i] += 1
                else:
                    self.stay_out[i] += 1
        # append info for the new node; for just enter, look at column is basic
        for i in range(len_actions):
            self.just_left.append(0)
            self.stay_out.append(0)
            self.stay_in.append(0)
        for i in range(len_actions):
            if self.ColumnIs_Basic[-len_actions+i] == 1:
                self.just_enter.append(1)
            else:
                self.just_enter.append(0)
        self.objVal_history.append(self.mainModel.objVal)
        self.node_count_history.append(0)
        dual_variables = np.array(
            [constraint.pi for constraint in self.mainModel.getConstrs()])
        # get the rc for all the columns
        self.RC = []
        for pattern in patterns:
            rc = 1-sum(dual_variables[i] * pattern[i]
                       for i in range(len(dual_variables)))
            self.RC.append(rc)
        self.Shadow_Price = dual_variables
        for i in range((self.n)):
            self.c[i].obj = dual_variables[i]
        self.subModel.Params.PoolSolutions = 10
        self.subModel.Params.PoolSearchMode = 2
        self.subModel.setAttr(GRB.Attr.ModelSense, -1)
        self.subModel.optimize()
        nSolutions = self.subModel.SolCount
        columns_to_select = []
        reduced_costs = []
        if (nSolutions >= 1):
            for i in range(nSolutions):
                sol = []
                self.subModel.setParam(GRB.Param.SolutionNumber, i)
                for e in range(self.n):
                    sol.append(int(self.c[e].Xn))
                rc = 1-sum(dual_variables[i] * sol[i]
                           for i in range(len(dual_variables)))
                if abs(rc - 0) <= 10**-4:
                    continue
                if rc >= 0:
                    continue
                columns_to_select.append(sol)
                reduced_costs.append(rc)
        done = self.subModel.objVal < 1+1e-5
        reward = self.alpha_obj_weight * \
            (self.objVal_history[-2] - self.objVal_history[-1]
             )/self.objVal_history[0]  # normalization term
        if Train:
            reward = reward-self.beta * \
                (len(action)-sum(self.useful_pattern[-len(action):]))
        if done:
            is_done = True
        else:
            next_available_action = (columns_to_select, reduced_costs)
            self.available_action = next_available_action
        return reward, is_done

import os, sys, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir) 
sys.path.insert(0,parentdir) 
import unittest
import cvxpy as cp
import numpy as np
import fisherVerifier
import fisherMarket

np.set_printoptions(formatter={'float': '{: 0.3f}'.format})


class MyTestCase(unittest.TestCase):
    def testExample1(self):
        """
        Example in the Eisenberg-Gale paper:
        Consensus of Subjective Probabilities: The Pari-Mutuel Method on JSTOR
        https://www.jstor.org/stable/2237130?casa_token=O576M6_fWzsAAAAA:QqMHo9vTXEQa1Q7X-dUkPDaNSneFLzhTmChI6XWKVFMgh6GE8iwI4x-vmezbE1fe4-KDKcsVPJBK1OmcSCgIVYmQ8Uv2SljiTxTp3Ur264Pwmn8Kzp0g&socuuid=8a7880bd-edae-4c49-b540-73ea19e69fba&socplat=twitter
        """
        valuations = np.array([[10.85, 10.5], [0.5, 0.5]])
        budgets = np.array([1, 1])
        m1 = fisherMarket.FisherMarket(valuations, budgets)
        X, p = m1.solveMarket("linear", printResults=False)
        print(X)
        print(p)

    def testMarkets(self, num_buyers=15, num_goods=20, num_trials=10, tolerance=1e-1):
        """
        Test num_trials many i.i.d. random markets with num_buyers many consumers and num_goods many goods.
        For each market, compute its equilibria as a fisher (linear) market. Then, fixing the equilibria price vector,
        compute for each consumer its utility maximizing bundle. The utility of the later computation should equal
        the utility of the equilibria computation. We check equality up to given tolerance.
        :param num_buyers: the number of consumers. Default is 20.
        :param num_goods: the number of goods. Default is 20.
        :param num_trials: the number of random, i.i.d. markets.
        """
        for trial in range(num_trials):
            try:
                # Matrix of valuations: |buyers| x |goods|
                valuations = np.random.rand(num_buyers, num_goods)*100

                # Budgets of buyers: |buyers|
                budgets = np.random.rand(num_buyers)*10

                print(f"\n ********** Linear Utility Tests **********")
                ################ Linear #################

                # Create Market
                market = fisherMarket.FisherMarket(valuations, budgets)
                X, p = market.solveMarket("linear", printResults=False)
                print(f"\n ***** Trial {trial} *****")
                
                
                consumer_eq_utility = np.sum(valuations*X, axis = 1)

                for buyer in range(num_buyers):
                    budget = budgets[buyer]
                    valuation = valuations[buyer]
                    max_u = fisherVerifier.getIndirectUtil(valuation, p, budget, utility = "linear")
                    error = abs(max_u - consumer_eq_utility[buyer])
                    print(f"{buyer}'s consumer error is {error}")
                    self.assertLess(error, tolerance*max_u)


                print(f"\n ********** Quasilinear Utility Tests **********")

                ################ Quasiinear #################

                # Create Market
                market = fisherMarket.FisherMarket(valuations, budgets)
                X, p = market.solveMarket("quasilinear", printResults=False)
                print(f"\n ***** Trial {trial} *****")
                
                
                consumer_eq_utility = np.sum((valuations- p)*X, axis = 1)
                print(consumer_eq_utility.shape)
                for buyer in range(num_buyers):
                    budget = budgets[buyer]
                    valuation = valuations[buyer]
                    max_u = fisherVerifier.getIndirectUtil((valuation - p), p, budget, utility = "linear")
                    error = abs(max_u - consumer_eq_utility[buyer])
                    print(f"{buyer}'s consumer error is {error}")
                    self.assertLess(error, tolerance*max_u)

                print(f"\n ********** Leontief Utility Tests **********")

            ################ Leontief #################

                # Create Market
                market = fisherMarket.FisherMarket(valuations, budgets)
                X, p = market.solveMarket("leontief", printResults=False)
                print(f"\n ***** Trial {trial} *****")
                
                
                consumer_eq_utility = np.min(X/valuations, axis = 1)

                for buyer in range(num_buyers):
                    budget = budgets[buyer]
                    valuation = valuations[buyer]
                    max_u = fisherVerifier.getIndirectUtil(valuation, p, budget, utility = "leontief")
                    error = abs(max_u - consumer_eq_utility[buyer])
                    print(f"{buyer}'s consumer error is {error}")
                    self.assertLess(error, tolerance)


         
                print(f"\n ********** CES Utility Tests **********")

                ################ CES #################
                rho = np.random.choice([0.3, 0.4, 0.5, 0.6, 0.7])
                # Create Market
                X, p = market.solveMarket("ces", printResults=False, rho = rho)

                
                consumer_eq_utility = np.sum(valuations*np.power(X, rho), axis = 1)**(1/rho)
                for buyer in range(num_buyers):
                    budget = budgets[buyer]
                    valuation = valuations[buyer]
                    max_x = fisherVerifier.getMarshallianDemand(valuation, p, budget, utility = "ces", rho = rho)
                    max_u = (valuation.T @ np.power(max_x, rho))**(1/rho)
                    error = abs(max_u - consumer_eq_utility[buyer])
                    print(f"{buyer}'s util1 {max_u} other util {consumer_eq_utility[buyer]}")
                    print(f"{buyer}'s consumer error is {error}")
                    self.assertLess(error, tolerance)


            except cp.error.SolverError:
                print("IGNORING")


if __name__ == '__main__':
    unittest.main()

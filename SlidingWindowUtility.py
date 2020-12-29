# importing all required modules
import warnings
warnings.filterwarnings("ignore")
import argparse
from PhiRelevance.PhiUtils import phiControl,phi
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
import math
from SmoteRSampling import SmoteR
from RelevanceBasedOverSampling import RelevanceOverSampling


def getParser():
    """
    :return: parser object containing all user arguments
    """


    parser = argparse.ArgumentParser(description="Sliding Window PhiRelevance Based Regression",
                                     usage="\n * example usage-1:             python SlidingWindowUtility.py -dataFile ./datasets/bike-sharing/hour.csv -outputLabel cnt -method extremes -methodLevel high -coef 0.05 \n * example usage-2:             python SlidingWindowUtility.py -dataFile ./datasets/bike-sharing/hour.csv -outputLabel cnt -method range -controlPoints [[1,1,0],[2,1,0],[3,1,1]]\n ")
    parser.add_argument("-dataFile",
                        dest="data_file",
                        help="Dataset filename with location")
    parser.add_argument("-outputLabel",
                        dest="output_label",
                        help="column in the data set representing the final outcome")
    parser.add_argument("-dateLabel",
                        dest="date_label",
                        help="column in the data set representing date field for sliding window")
    parser.add_argument("-method",
                        default="extremes",
                        dest="method",
                        help="A string representing method name ('extremes','range'). default is extremes")
    parser.add_argument("-methodLevel",
                        default="high",
                        dest="method_level",
                        help="This is only for extremes method type. method level can be ('high','low','both') and default is 'high'")
    parser.add_argument("-controlPoints",
                        default=[],
                        dest="control_points",
                        help="2D List of size (nx2) or (nx3). this field must be provided for range type and default in empty list")
    parser.add_argument("-coef",
                        default="0.05",
                        dest="coef",
                        help="coefficient is a Float value and used for extremes. default is 0.05")
    parser.add_argument("-slidingWindowSize",
                        default="10",
                        dest="sliding_window_size",
                        help="slidingWindowSize is an integer, describes the size of window to be considered for dataset")
    parser.add_argument("-dynamicUtility",
                        default="1",
                        dest="dynamic_utility",
                        help="dynamicUtility is an boolean variable. 1: Use Dynamic PhiRelevance, 0: Static PhiRelevance")
    parser.add_argument("-regressionAlgo",
                        default="1",
                        dest="regression_algo",
                        help="Integer representing regression algorithm to use. 1: Random Forest, 2: Decision Tree, 3: Support Vector Machine 4: K Nearest Neighbour")
    parser.add_argument("-sampling",
                        default="0",
                        dest="sampling",
                        help="Integer representing sampling algorithm to use for data. 1: SmoteR, 2: RevelanceBasedOversampling 0: No Samplinge")

    return parser


def plotRelevance(y,relevance,start_date,end_date):
    """

    :param y: Continous target variable
    :param relevance: Relevance Phi values associated with y data
    :param start_date: Window Start date
    :param end_date: Window end date
    :return:
    """
    plt.plot(y,relevance,'ro')
    plt.title("PhiRelevance from: "+str(start_date)+" and till: "+str(end_date))
    plt.xlabel('Class Labels')
    plt.ylabel('relevance')
    plt.show()

def plotHistogram(dataframe,label):
    """

    :param dataframe: dataframe object of data
    :param label: name of continous target variable
    :return: None
    """
    plt.hist(dataframe[label], bins=10)
    plt.show()

def callUtilityFunctions(data,method,extrType, controlPts, coef , start_date = '',end_date=''):
    """

    :param data: Continous Target Variable
    :param method: method type ("extremes", "range")
    :param extrType: required argument for extremes method ("low", "high", "both")
    :param controlPts: required argument for range method (nx2 or nx3 matrix for contruct relevance)
    :param coef: coefficient for extreme method
    :param start_date: Window Start Date
    :param end_date: Window End Date
    :return:
        list
            yphi (phi values for target varaibles)
        list
            ydphi (first derivative phi values for target varaibles)
        list
            yddphi (second derivative phi values for target varaibles)
    """

    if(method=="extremes"):
        controlPts, npts = phiControl(data,method,extrType, list(controlPts), coef)
    else:
        controlPts, npts = phiControl(data,method,extrType, list(controlPts), coef)
    # print("\n------------------------------------------")
    # print("Control Points", list(controlPts))
    # print("npts:", npts)
    # print("\n------------------------------------------")

    if(controlPts == -1 and npts == -1):
        print("Invalid Parameters")
        return [],[],[]

    yPhi, ydPhi, yddPhi = phi(data, list(controlPts), npts, 'extremes')

    return  yPhi, ydPhi, yddPhi



def slidingWindow(data,label,date_label,window):
    """
    Function for Window Sliding
    :param data: dataframe object for complete data
    :param label: column name for target variable
    :param date_label: date label in dataset
    :param window: window size
    :return: returns window data, start date and end date
    """

    print("\n******************************************")
    print("Getting Ready for sliding window")
    print("\n******************************************")

    data[date_label] = pd.to_datetime(data[date_label], format='%Y-%m-%d')
    data = data.sort_values(by=[date_label])
    print("\n------------------------------------------")
    print("The data set has data starting from: "+str(data[date_label].iloc[0])+"and ending from: "+str(data[date_label].iloc[-1]))

    r = pd.date_range(start=data[date_label].iloc[0], end=data[date_label].iloc[-1])
    L = [(d.strftime('%Y-%m-%d'), (d + pd.Timedelta(window-1, 'd')).strftime('%Y-%m-%d')) for d in r]
    print("\n------------------------------------------")
    print(L)
    print(len(L))
    print("\n------------------------------------------")

    for i in L:
        window_data_indicies = (data[date_label] >= i[0]) & (data[date_label] <= i[1])
        window_data = data.loc[window_data_indicies]
        yield window_data,i[0],i[1]




if __name__ == '__main__':

    print("Reading and Analysing Command Line Arguments\n")
    try:
        parser = getParser()
        args = parser.parse_args()
    except:
        print("\nTry *** python SlidingWindowUtility.py -h or --help *** for more details.")
        exit(0)

    print(args)
    data_file = args.data_file
    label = args.output_label
    date_label = args.date_label
    method = args.method
    extrType = args.method_level
    controlPts = list(args.control_points)
    coef = float(args.coef)
    sampling = int(args.sampling)

    sliding_window_size = int(args.sliding_window_size)
    dynamic_utility = int(args.dynamic_utility)
    regression_algo = int(args.regression_algo)


    data = pd.read_csv(data_file)

    y_true = []
    y_pred = []
    phi_values = []


    yPhi, ydPhi, yddPhi = [], [] , []
    static_utility_cal_once = 0


    for window_data, start_date, end_date in slidingWindow(data, label, date_label, sliding_window_size):


        window_data = window_data.drop(columns = [date_label])

        train = window_data.head(-1)
        test = window_data.tail(1)

        if(train.shape[0] <= 4):
            continue


        X_train,y_train,X_test,y_test = train.drop(columns = [label]).values.tolist(), train[label].tolist(), test.drop(columns = [label]).values.tolist(), test[label].tolist()
        y_test = y_test[0]

        window_phi = []
        if (dynamic_utility == 1):
            yPhi, ydPhi, yddPhi = callUtilityFunctions(y_train, method, extrType, controlPts, coef, start_date,
                                                       end_date)
            phi_values.append(np.interp(y_test, y_train, yPhi))
            window_phi = yPhi


            plotRelevance(y_train, window_phi, start_date, end_date)


        elif (dynamic_utility == 0):
            if (static_utility_cal_once == 0):
                yPhi, ydPhi, yddPhi = callUtilityFunctions(data[label], method, extrType, controlPts, coef, start_date,
                                                           end_date)
                static_utility_cal_once = 1
            phi_values.append(yPhi[data[label].values.tolist().index(y_test)])

            for y_instance in train[label].values:
                # window_phi.append(yPhi[data[label].values.tolist().index(y_instance)])
                window_phi_indices = data.index[data[label] == y_instance].tolist()
                window_phi.append(yPhi[window_phi_indices[-1]])


            plotRelevance(train[label].values, window_phi, start_date, end_date)



        # print("before sampling")
        # print("Shape of Window data "+str(train.shape))
        # plotHistogram(train,label)

        if (sampling == 1):
            s = SmoteR(train, target='temp',relevance = np.asarray(window_phi) , categorical_col=data.columns)
            train = s

        elif (sampling == 2):
            s = RelevanceOverSampling(train, target='temp', size = int(train.shape[0]/2), relevance= window_phi, categorical_col=data.columns)
            train = s

        # print("after sampling")
        # print("Shape of Window data "+str(train.shape))
        # plotHistogram(train, label)


        X_train, y_train, X_test, y_test = train.drop(columns=[label]).values.tolist(), train[label].tolist(), test.drop(columns=[label]).values.tolist(), test[label].tolist()
        y_test = y_test[0]



        if(regression_algo == 1):
            regr = RandomForestRegressor(max_depth=100, random_state=0)
        elif(regression_algo == 2):
            regr =  DecisionTreeRegressor(max_depth=100, random_state=0)
        elif (regression_algo == 3):
            regr = SVR(kernel="rbf")
        elif (regression_algo == 4):
            regr = KNeighborsRegressor(n_neighbors=3)


        regr.fit(X_train,y_train)
        predicted = regr.predict(X_test)

        y_true.append(y_test)
        y_pred.append(predicted[0])


    RMSE_Phi = 0
    RMSE = 0


    y_diff = [(y_true[i]-y_pred[i])**2 for i in range(len(y_true))]

    RMSE = math.sqrt(sum(y_diff)/len(y_diff))

    y_diff_phi = [y_diff[i]*phi_values[i] for i in range(len(y_diff))]

    RMSE_Phi = math.sqrt(sum(y_diff_phi) / len(y_diff_phi))


    print("Original RMSE "+ str(RMSE))

    print("RMSE with PHI " + str(RMSE_Phi))

    File_object = open("results.txt", "a+")

    s = "\n---------------------------"
    s += method +" "+extrType+" dynamic: "+str(dynamic_utility)+" regression algo: "+str(regression_algo)+" sampling:"+str(sampling)+"\n"
    s += "Original RMSE "+ str(RMSE) +"\n"
    s += "RMSE with PHI " + str(RMSE_Phi) + "\n"
    s += "---------------------------\n"

    File_object.write(s)
    File_object.close()

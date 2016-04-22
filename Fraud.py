from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from skimage import measure
from sklearn import svm
import matplotlib.font_manager
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import os


def list_maker(in_list):
    out_list = list()
    for each_val in in_list:
        out_list.append([x for x in (each_val.strip().replace(" ", "")).split(",")])
    return out_list

name = input("Enter Your name: ")
for file in os.listdir(os.getcwd()):
    if file.startswith(name.upper()):
        name = file
try:
    open_file = open(name, 'r')
except IOError:
    print("Error: The file corresponding to your name was not found.")
else:
    if os.stat(name).st_size <= 0:
        print("The file is empty")
    else:
        with open(name, 'r') as file:
            number_Of_examples = int(file.readline())
            name_Of_attributes = list_maker(file.readline().split(","))
            examples = list_maker(file.readlines())
            SPACE_SAMPLING_POINTS = 100
            TRAIN_POINTS = 100
            # Define the size of the space which is interesting for the example
            X_MIN = -10
            X_MAX = 10
            Y_MIN = -10
            Y_MAX = 10
            Z_MIN = -10
            Z_MAX = 10

            # Makes the line space that is required to draw the graph
            xx, yy, zz = np.meshgrid(np.linspace(X_MIN, X_MAX, SPACE_SAMPLING_POINTS),
                                     np.linspace(Y_MIN, Y_MAX, SPACE_SAMPLING_POINTS),
                                     np.linspace(Z_MIN, Z_MAX, SPACE_SAMPLING_POINTS))
            # Needs to be changed
            # Generation of random points
            X = 0.3 * np.random.randn(TRAIN_POINTS, 3)
            X_train = np.r_[X + 2, X - 2, X + [2, 2, 0]]

            X = 0.3 * np.random.randn(20, 3)
            X_test = np.r_[X + 2, X - 2, X + [2, 2, 0]]

            X_outliers = np.random.uniform(low=-4, high=4, size=(20, 3))

            # Defines the svm-OneClassSVM with the rbf regression
            clf = svm.OneClassSVM(nu=0.1, kernel="rbf", gamma=0.1)
            clf.fit(X_train)

            # predicting the values
            y_pred_train = clf.predict(X_train)
            y_pred_test = clf.predict(X_test)
            y_pred_outliers = clf.predict(X_outliers)

            # get the error size
            n_error_train = y_pred_train[y_pred_train == -1].size
            n_error_test = y_pred_test[y_pred_test == -1].size
            n_error_outliers = y_pred_outliers[y_pred_outliers == 1].size

            # Distance of the samples X, Y, Z to the sperating hyperplane
            Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel(), zz.ravel()])
            Z = Z.reshape(xx.shape)

            plt.title("Novelty Detection")
            ax = plt.gca(projection='3d')

            # draw the points on the 3D graph
            b1 = ax.scatter(X_train[:, 0], X_train[:, 1], X_train[:, 2], c='white')
            b2 = ax.scatter(X_test[:, 0], X_test[:, 1], X_test[:, 2], c='green')
            c = ax.scatter(X_outliers[:, 0], X_outliers[:, 1], X_outliers[:, 2], c='red')

            # Signifies the Learning Curve
            # IMP it should be better as the data set gets better
            verts, faces = measure.marching_cubes(Z, 0)
            verts = verts * \
                [X_MAX - X_MIN, Y_MAX - Y_MIN, Z_MAX - Z_MIN] / SPACE_SAMPLING_POINTS
            verts = verts + [X_MIN, Y_MIN, Z_MIN]
            mesh = Poly3DCollection(verts[faces], facecolor='orange', edgecolor='gray', alpha=0.3)
            ax.add_collection3d(mesh)

            ax.set_xlim((-10, 10))
            ax.set_ylim((-10, 10))
            ax.set_zlim((-10, 10))
            ax.axis('tight')

            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.set_zlabel("Z")
            ax.legend([mpatches.Patch(color='orange', alpha=0.3), b1, b2, c],
                      ["learned frontier", "training observations",
                       "new regular observations", "new abnormal observations"],
                      loc="lower left",
                      prop=matplotlib.font_manager.FontProperties(size=11))
            ax.set_title(
                "error train: %d/200 ; errors novel regular: %d/40 ; "
                "errors novel abnormal: %d/40"
                % (n_error_train, n_error_test, n_error_outliers))

            plt.show()


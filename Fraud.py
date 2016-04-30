from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from skimage import measure
from sklearn import svm
import matplotlib.font_manager
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
import os

# Function that return a list of each lists
def list_maker(in_list):
    out_list = list()
    for each_val in in_list:
        out_list.append([x for x in (each_val.strip().replace(" ", "")).split(",")])
    return out_list
# Input is taken as the name
name = input("Enter Your name: ")
# transaction = input("Enter Your transaction: ")


for file in os.listdir(os.getcwd()):
    # Search if the files that start with the input name
    if file.startswith(name.upper()):
        name = file
    # if file.startswith(transaction.upper()):
    #     transaction = file
try:
    # Converts the text or input file in a numpy.array
    open_file = np.loadtxt(name)


except IOError:
    print("Error: The file corresponding to your name was not found.")
else:
    # if the input file has no values
    if os.stat(name).st_size <= 0: #or os.stat(transaction).st_size <= 0:
        print("The file is empty")
    else:
        with open(name, 'r') as file:

            SPACE_SAMPLING_POINTS = 100
            TRAIN_POINTS = 100
            #Define the size of the space which is interesting for the example
            X_MIN = -20
            X_MAX = 20
            Y_MIN = -20
            Y_MAX = 20
            Z_MIN = -20
            Z_MAX = 20

            # Makes the line space that is required to draw the graph
            xx, yy, zz = np.meshgrid(np.linspace(X_MIN, X_MAX, SPACE_SAMPLING_POINTS),
                                     np.linspace(Y_MIN, Y_MAX, SPACE_SAMPLING_POINTS),
                                     np.linspace(Z_MIN, Z_MAX, SPACE_SAMPLING_POINTS))

            X_train = open_file

            # Dimensional Reduction from 15 to 3
            pca = PCA(n_components=3)
            pca.fit(X_train)
            X_train = pca.transform(X_train)

            # X_test = pca.explained_variance_ratio_

            # Defines the svm-OneClassSVM with the rbf regression
            clf = svm.OneClassSVM(nu=0.1, kernel="rbf", gamma=0.1)
            clf.fit(X_train)

            # predicting the values
            y_pred_train = clf.predict(X_train)

            # get the error size
            n_error_train = y_pred_train[y_pred_train == -1].size

            # Distance of the samples X, Y, Z to the sperating hyperplane
            Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel(), zz.ravel()])
            Z = Z.reshape(xx.shape)

            plt.title("Novelty Detection")
            ax = plt.gca(projection='3d')

            # draw the points on the 3D graph
            b1 = ax.scatter(X_train[:, 0], X_train[:, 1], X_train[:, 2], c='white')

            # Signifies the Learning Curve
            # IMP it should be better as the data set gets better
            verts, faces = measure.marching_cubes(Z, 0)
            verts = verts * \
                [X_MAX - X_MIN, Y_MAX - Y_MIN, Z_MAX - Z_MIN] / SPACE_SAMPLING_POINTS
            verts = verts + [X_MIN, Y_MIN, Z_MIN]
            mesh = Poly3DCollection(verts[faces], facecolor='orange', edgecolor='gray', alpha=0.3)
            ax.add_collection3d(mesh)

            ax.set_xlim((-20, 20))
            ax.set_ylim((-20, 20))
            ax.set_zlim((-20, 20))
            ax.axis('tight')

            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.set_zlabel("Z")
            # ax.legend([mpatches.Patch(color='orange', alpha=0.3), b1, b2, c],
            ax.legend([mpatches.Patch(color='orange', alpha=0.3), b1],
                      ["learned frontier", "training observations",
                       "new regular observations", "new abnormal observations"],
                      loc="lower left",
                      prop=matplotlib.font_manager.FontProperties(size=11))
            ax.set_title(
                "error train: %d/200 ; "
                % n_error_train)
            plt.show()


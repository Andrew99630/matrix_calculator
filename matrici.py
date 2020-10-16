import sys
import numpy as np
from PyQt5 import QtWidgets, QtCore
from design import Ui_MainWindow


class Matrix(Ui_MainWindow, QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)

    def size_change(self, matrix_name):
        self.hidespin(matrix_name)
        row_number = getattr(self, 'spin_1%s' % matrix_name).value()
        column_number = getattr(self, 'spin_2%s' % matrix_name).value()
        for r in range(1, row_number + 1):
            for c in range(1, column_number + 1):
                getattr(self, 'spin_%s%s%s' % (r, c, matrix_name)).show()
        getattr(self, 'label_r%s' % matrix_name)

    def clear(self, matrix_name):
        for row in range(1, 6):
            for col in range(1, 6):
                getattr(self, 'spin_%s%s%s' % (row, col, matrix_name)).setValue(0)
        getattr(self, 'label_r%s' % matrix_name)

    def transposition(self, matrix_name):
        self.hidespin(matrix_name)
        row_number = getattr(self, 'spin_1%s' % matrix_name).value()
        column_number = getattr(self, 'spin_2%s' % matrix_name).value()
        A = np.zeros((row_number, column_number))
        for r in range(1, row_number + 1):
            for c in range(1, column_number + 1):
                A[r - 1, c - 1] = getattr(self, 'spin_%s%s%s' % (r, c, matrix_name)).value()
        A = np.transpose(A)
        getattr(self, 'spin_1%s' % matrix_name).setValue(column_number)
        getattr(self, 'spin_2%s' % matrix_name).setValue(row_number)
        row_number = getattr(self, 'spin_1%s' % matrix_name).value()
        column_number = getattr(self, 'spin_2%s' % matrix_name).value()
        for r in range(1, row_number + 1):
            for c in range(1, column_number + 1):
                getattr(self, 'spin_%s%s%s' % (r, c, matrix_name)).show()
                getattr(self, 'spin_%s%s%s' % (r, c, matrix_name)).setValue(A[r - 1, c - 1])

    def null_matrix(self, matrix_name):
        getattr(self, 'label_r%s' % matrix_name).hide()
        row_number = getattr(self, 'spin_1%s' % matrix_name).value()
        column_number = getattr(self, 'spin_2%s' % matrix_name).value()
        for row in range(1, row_number + 1):
            for col in range(1, column_number + 1):
                getattr(self, 'spin_%s%s%s' % (row, col, matrix_name)).setValue(0)

    def identity_matrix(self, matrix_name):
        getattr(self, 'label_r%s' % matrix_name).hide()
        row_number = getattr(self, 'spin_1%s' % matrix_name).value()
        column_number = getattr(self, 'spin_2%s' % matrix_name).value()
        if row_number != column_number or getattr(self, 'spin_12%s' % matrix_name).isHidden():
            QtWidgets.QMessageBox.critical(self, 'Ошибка', 'Данная операция выполнима только для квадратной матрицы.',
                                           defaultButton=QtWidgets.QMessageBox.Ok)
        else:
            self.null_matrix(matrix_name)
            for row in range(1, row_number + 1):
                for col in range(1, column_number + 1):
                    if row == col:
                        getattr(self, 'spin_%s%s%s' % (row, col, matrix_name)).setValue(1)

    def diagonal_matrix(self, matrix_name):
        getattr(self, 'label_r%s' % matrix_name).hide()
        row_number = getattr(self, 'spin_1%s' % matrix_name).value()
        column_number = getattr(self, 'spin_2%s' % matrix_name).value()
        if row_number != column_number or getattr(self, 'spin_12%s' % matrix_name).isHidden():
            QtWidgets.QMessageBox.critical(self, 'Ошибка', 'Данная операция выполнима только для квадратной матрицы.',
                                           defaultButton=QtWidgets.QMessageBox.Ok)
        else:
            for row in range(1, row_number + 1):
                for col in range(1, column_number + 1):
                    if row != col:
                        getattr(self, 'spin_%s%s%s' % (row, col, matrix_name)).setValue(0)

    def upper_matrix(self, matrix_name):  # upper triangular matrix
        getattr(self, 'label_r%s' % matrix_name).hide()
        row_number = getattr(self, 'spin_1%s' % matrix_name).value()
        column_number = getattr(self, 'spin_2%s' % matrix_name).value()
        if row_number != column_number or getattr(self, 'spin_12%s' % matrix_name).isHidden():
            QtWidgets.QMessageBox.critical(self, 'Ошибка', 'Данная операция выполнима только для квадратной матрицы.',
                                           defaultButton=QtWidgets.QMessageBox.Ok)
        else:
            for row in range(1, row_number + 1):
                for col in range(1, column_number + 1):
                    if row != col and col < row:
                        getattr(self, 'spin_%s%s%s' % (row, col, matrix_name)).setValue(0)

    def lower_matrix(self, matrix_name):  # lower triangular matrix
        getattr(self, 'label_r%s' % matrix_name).hide()
        row_number = getattr(self, 'spin_1%s' % matrix_name).value()
        column_number = getattr(self, 'spin_2%s' % matrix_name).value()
        if row_number != column_number or getattr(self,'spin_12%s' % matrix_name).isHidden():
            QtWidgets.QMessageBox.critical(self, 'Ошибка', 'Данная операция выполнима только для квадратной матрицы.',
                                           defaultButton=QtWidgets.QMessageBox.Ok)
        else:
            for row in range(1, row_number + 1):
                for col in range(1, column_number + 1):
                    if row != col and col > row:
                        getattr(self, 'spin_%s%s%s' % (row, col, matrix_name)).setValue(0)

    def addition(self, matrix_name):
        getattr(self, 'label_r%s' % matrix_name).hide()
        row_number = getattr(self, 'spin_1%s' % matrix_name).value()
        column_number = getattr(self, 'spin_2%s' % matrix_name).value()
        number = getattr(self, 'spin_p3%s' % matrix_name).value()
        for row in range(1, row_number + 1):
            for col in range(1, column_number + 1):
                getattr(self, 'spin_%s%s%s' % (row, col, matrix_name)).setValue(
                    getattr(self, 'spin_%s%s%s' % (row, col, matrix_name)).value() + number)

    def multiplication(self, matrix_name):
        getattr(self, 'label_r%s' % matrix_name).hide()
        row_number = getattr(self, 'spin_1%s' % matrix_name).value()
        column_number = getattr(self, 'spin_2%s' % matrix_name).value()
        multiplier = getattr(self, 'spin_p1%s' % matrix_name).value()
        for row in range(1, row_number + 1):
            for col in range(1, column_number + 1):
                getattr(self, 'spin_%s%s%s' % (row, col, matrix_name)).setValue(
                    getattr(self, 'spin_%s%s%s' % (row, col, matrix_name)).value() * multiplier)

    def pow(self, matrix_name):
        row_number = getattr(self, 'spin_1%s' % matrix_name).value()
        column_number = getattr(self, 'spin_2%s' % matrix_name).value()
        if row_number != column_number or getattr(self, 'spin_12%s' % matrix_name).isHidden():
            QtWidgets.QMessageBox.critical(self, 'Ошибка', 'Данная операция выполнима только для квадратной матрицы.',
                                           defaultButton=QtWidgets.QMessageBox.Ok)
        else:
            getattr(self, 'label_r%s' % matrix_name).hide()
            power = getattr(self, 'spin_p2%s' % matrix_name).value()
            A = np.zeros((row_number, column_number))
            for r in range(1, row_number + 1):
                for c in range(1, column_number + 1):
                    A[r - 1, c - 1] = getattr(self, 'spin_%s%s%s' % (r, c, matrix_name)).value()
            A = np.linalg.matrix_power(A, power)
            for r in range(1, row_number + 1):
                for c in range(1, column_number + 1):
                    getattr(self, 'spin_%s%s%s' % (r, c, matrix_name)).setValue(A[r - 1, c - 1])

    def determinant(self, matrix_name):
        row_number = getattr(self, 'spin_1%s' % matrix_name).value()
        column_number = getattr(self, 'spin_2%s' % matrix_name).value()
        if row_number != column_number or getattr(self, 'spin_12%s' % matrix_name).isHidden():
            QtWidgets.QMessageBox.critical(self, 'Ошибка', 'Данная операция выполнима только для квадратной матрицы.',
                                           defaultButton=QtWidgets.QMessageBox.Ok)
        else:
            A = np.zeros((row_number, column_number))
            for r in range(1, row_number + 1):
                for c in range(1, column_number + 1):
                    A[r - 1, c - 1] = getattr(self, 'spin_%s%s%s' % (r, c, matrix_name)).value()
            determinant = round(np.linalg.det(A), 3)
            getattr(self, 'label_r%s' % matrix_name).setText('Определитель данной матрицы равен: %s' % determinant)
            getattr(self, 'label_r%s' % matrix_name).show()

    def inverse_matrix(self, matrix_name):
        row_number = getattr(self, 'spin_1%s' % matrix_name).value()
        column_number = getattr(self, 'spin_2%s' % matrix_name).value()
        if row_number != column_number or getattr(self, 'spin_12%s' % matrix_name).isHidden():
            QtWidgets.QMessageBox.critical(self, 'Ошибка', 'Данная операция выполнима только для квадратной матрицы.',
                                           defaultButton=QtWidgets.QMessageBox.Ok)
        else:
            A = np.zeros((row_number, column_number))
            for r in range(1, row_number + 1):
                for c in range(1, column_number + 1):
                    A[r - 1, c - 1] = getattr(self, 'spin_%s%s%s' % (r, c, matrix_name)).value()
            if np.linalg.det(A) == 0:
                QtWidgets.QMessageBox.critical(self, 'Ошибка', 'Определитель матрицы равен 0. Обратной матрицы не существует.',
                                               defaultButton=QtWidgets.QMessageBox.Ok)
            else:
                A = np.linalg.inv(A)
                for r in range(1, row_number + 1):
                    for c in range(1, column_number + 1):
                        getattr(self, 'spin_%s%s%s' % (r, c, matrix_name)).setValue(A[r - 1, c - 1])

    def rank(self, matrix_name):
        row_number = getattr(self, 'spin_1%s' % matrix_name).value()
        column_number = getattr(self, 'spin_2%s' % matrix_name).value()
        A = np.zeros((row_number, column_number))
        for r in range(1, row_number + 1):
            for c in range(1, column_number + 1):
                A[r - 1, c - 1] = getattr(self, 'spin_%s%s%s' % (r, c, matrix_name)).value()
        getattr(self, 'label_r%s' % matrix_name).setText('Ранг данной матрицы равен %s' % np.linalg.matrix_rank(A))
        getattr(self, 'label_r%s' % matrix_name).show()

    def swap(self):
        row_A = self.spin_1A.value()
        col_A = self.spin_2A.value()
        row_B = self.spin_1B.value()
        col_B = self.spin_2B.value()
        A1 = np.zeros((row_A, col_A))  # Создание нулевой матрицы для последующего заполнения
        A2 = np.zeros((row_B, col_B))
        for r in range(1, row_A + 1):
            for c in range(1, col_A + 1):
                A1[r - 1, c - 1] = getattr(self, 'spin_%s%sA' % (r, c)).value()
        for r in range(1, row_B + 1):
            for c in range(1, col_B + 1):
                A2[r - 1, c - 1] = getattr(self, 'spin_%s%sB' % (r, c)).value()
        self.spin_1A.setValue(row_B)
        self.spin_2A.setValue(col_B)
        self.spin_1B.setValue(row_A)
        self.spin_2B.setValue(col_A)
        self.hidespin('A')
        self.hidespin('B')
        for r in range(1, self.spin_1A.value() + 1):
            for c in range(1, self.spin_2A.value() + 1):
                getattr(self, 'spin_%s%sA' % (r, c)).show()
        for r in range(1, self.spin_1B.value() + 1):
            for c in range(1, self.spin_2B.value() + 1):
                getattr(self, 'spin_%s%sB' % (r, c)).show()
        for r in range(1, self.spin_1A.value() + 1):
            for c in range(1, self.spin_2A.value() + 1):
                getattr(self, 'spin_%s%sA' % (r, c)).setValue(A2[r - 1, c - 1])
        for r in range(1, self.spin_1B.value() + 1):
            for c in range(1, self.spin_2B.value() + 1):
                getattr(self, 'spin_%s%sB' % (r, c)).setValue(A1[r - 1, c - 1])
        self.label_rA.hide()
        self.label_rB.hide()

    def add_dif(self, operation):
        row_A = self.spin_1A.value()
        column_A = self.spin_2A.value()
        row_B = self.spin_1A.value()
        column_B = self.spin_2A.value()
        if row_A != row_B or column_A != column_B:
            QtWidgets.QMessageBox.critical(self, 'Ошибка', 'Для выполнения операции создайте две матрицы одинакового размера.',
                                           defaultButton=QtWidgets.QMessageBox.Ok)
        else:
            A = np.zeros((row_A, column_A))
            B = np.zeros((row_B, column_B))
            for r in range(1, row_A + 1):
                for c in range(1, column_A + 1):
                    A[r - 1, c - 1] = getattr(self, 'spin_%s%sA' % (r, c)).value()
            for r in range(1, row_B + 1):
                for c in range(1, column_B + 1):
                    B[r - 1, c - 1] = getattr(self, 'spin_%s%sB' % (r, c)).value()
            C = 0
            if operation == 'add':
                C = A + B
            elif operation == 'dif':
                C = A - B
            self.spin_1C.setValue(row_A)
            self.spin_2C.setValue(column_A)
            self.hidespin('C')
            for r in range(1, row_A + 1):
                for c in range(1, column_A + 1):
                    getattr(self, 'spin_%s%sC' % (r, c)).setValue(C[r - 1, c - 1])
                    getattr(self, 'spin_%s%sC' % (r, c)).show()

    def matrix_mult(self):
        row_A = self.spin_1A.value()
        column_A = self.spin_2A.value()
        row_B = self.spin_1B.value()
        column_B = self.spin_2B.value()
        if column_A != row_B:
            QtWidgets.QMessageBox.critical(self, 'Ошибка', 'Для перемножения матриц необходимо равенство\nчисла стобцов матрицы А и строк матрицы B.',
                                           defaultButton=QtWidgets.QMessageBox.Ok)
        else:
            A = np.zeros((row_A, column_A))
            B = np.zeros((row_B, column_B))
            for r in range(1, row_A + 1):
                for c in range(1, column_A + 1):
                    A[r - 1, c - 1] = getattr(self, 'spin_%s%sA' % (r, c)).value()
            for r in range(1, row_B + 1):
                for c in range(1, column_B + 1):
                    B[r - 1, c - 1] = getattr(self, 'spin_%s%sB' % (r, c)).value()
            C = np.dot(A, B)
            self.spin_1C.setValue(row_A)
            self.spin_2C.setValue(column_B)
            self.hidespin('C')
            for r in range(1, row_A + 1):
                for c in range(1, column_B + 1):
                    getattr(self, 'spin_%s%sC' % (r, c)).setValue(C[r - 1, c - 1])
                    getattr(self, 'spin_%s%sC' % (r, c)).show()

    def qr(self, matrix_name):
        row_number = getattr(self, 'spin_1%s' % matrix_name).value()
        column_number = getattr(self, 'spin_2%s' % matrix_name).value()
        if row_number != column_number or getattr(self, 'spin_12%s' % matrix_name).isHidden():
            QtWidgets.QMessageBox.critical(self, 'Ошибка', 'Данная операция выполнима только для квадратной матрицы.',
                                           defaultButton=QtWidgets.QMessageBox.Ok)
        else:
            A = np.zeros((row_number,  column_number))
            for r in range(1,  row_number + 1):
                for c in range(1, column_number + 1):
                    A[r - 1, c - 1] = getattr(self, 'spin_%s%s%s' % (r, c, matrix_name)).value()
            if np.linalg.det(A) == 0:
                QtWidgets.QMessageBox.critical(self, 'Ошибка', 'Определитель равен 0. QR-разложение невозможно.',
                                               defaultButton=QtWidgets.QMessageBox.Ok)
            else:
                Q, R = np.linalg.qr(A)
                self.hide_qr()
                for r in range(1, row_number + 1):
                    for c in range(1, column_number + 1):
                        getattr(self, 'label_%s%sq' % (r, c)).setNum(round(Q[r - 1, c - 1], 3))
                        self.label_q.setText('{} = Q x R, где Q = '.format(matrix_name))
                        self.label_q.show()
                        getattr(self, 'label_%s%sq' % (r, c)).show()
                for r in range(1, row_number + 1):
                    for c in range(1, column_number + 1):
                        getattr(self, 'label_%s%sr' % (r, c)).setNum(round(R[r - 1, c - 1], 3))
                        self.label_r.setText('R = ')
                        self.label_r.show()
                        getattr(self, 'label_%s%sr' % (r, c)).show()

    def choleskiy(self, matrix_name):
        row_number = getattr(self, 'spin_1%s' % matrix_name).value()
        column_number = getattr(self, 'spin_2%s' % matrix_name).value()
        A = np.zeros((row_number, column_number))
        for r in range(1, row_number + 1):
            for c in range(1, column_number + 1):
                A[r - 1, c - 1] = getattr(self, 'spin_%s%s%s' % (r, c, matrix_name)).value()
        if row_number == column_number > 0 and np.array_equal(A, np.transpose(A)) and np.linalg.det(A):
            L = np.linalg.cholesky(A)
            L_T = np.transpose(L)
            self.hide_qr()
            for r in range(1, row_number + 1):
                for c in range(1, column_number + 1):
                    getattr(self, 'label_%s%sq' % (r, c)).setNum(round(L[r - 1, c - 1], 3))
                    self.label_q.setText('{} = L x L^T, где L = '.format(matrix_name))
                    self.label_q.show()
                    getattr(self, 'label_%s%sq' % (r, c)).show()
                    getattr(self, 'label_%s%sr' % (r, c)).setNum(round(L_T[r - 1, c - 1], 3))
                    self.label_r.setText('LT = ')
                    self.label_r.show()
                    getattr(self, 'label_%s%sr' % (r, c)).show()

        else:
            QtWidgets.QMessageBox.critical(self, 'Ошибка', 'Матрица должна быть симметричной и положительно определённой.',
                                           defaultButton=QtWidgets.QMessageBox.Ok)

    def hidespin(self, matrix_name):
        for r in range(1, 6):
            for c in range(1, 6):
                getattr(self, 'spin_%s%s%s' % (r, c, matrix_name)).hide()
        getattr(self, 'label_r%s' % matrix_name)

    def hide_qr(self):
        qr = ['q', 'r']
        for r in range(1, 6):
            for c in range(1, 6):
                for k in qr:
                    getattr(self, 'label_%s%s%s' % (r, c, k)).hide()
        self.label_q.hide()
        self.label_r.hide()

 
if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    window = Matrix()
    window.setWindowFlags(QtCore.Qt.MSWindowsFixedSizeDialogHint)
    window.setFixedSize(1450, 840)
    window.setWindowTitle('Matrix calculator')
    window.show()
    sys.exit(app.exec_())

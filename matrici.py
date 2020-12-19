import sys

import numpy as np
from PyQt5 import QtWidgets, QtCore

from design import Ui_MainWindow


class Matrix(Ui_MainWindow, QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)

    def hide_cells(self, matrix_name):  # hide all cells
        for row in range(1, 6):
            for col in range(1, 6):
                getattr(self, 'spin_%s%s%s' % (row, col, matrix_name)).hide()
        getattr(self, 'label_r%s' % matrix_name)

    def size_change(self, matrix_name):
        print(matrix_name)
        self.hide_cells(matrix_name)
        row_number = getattr(self, 'spin_1%s' % matrix_name).value()
        column_number = getattr(self, 'spin_2%s' % matrix_name).value()
        for row in range(1, row_number + 1):
            for col in range(1, column_number + 1):
                getattr(self, 'spin_%s%s%s' % (row, col, matrix_name)).show()
        getattr(self, 'label_r%s' % matrix_name)

    def clear(self, matrix_name):
        for row in range(1, 6):
            for col in range(1, 6):
                getattr(self, 'spin_%s%s%s' % (row, col, matrix_name)).setValue(0)
        getattr(self, 'label_r%s' % matrix_name)

    def transposition(self, matrix_name):
        self.hide_cells(matrix_name)
        row_number = getattr(self, 'spin_1%s' % matrix_name).value()
        column_number = getattr(self, 'spin_2%s' % matrix_name).value()
        matrix = np.zeros((row_number, column_number))
        for row in range(1, row_number + 1):
            for col in range(1, column_number + 1):
                matrix[row - 1, col - 1] = getattr(self, 'spin_%s%s%s' % (row, col, matrix_name)).value()
        matrix = np.transpose(matrix)
        getattr(self, 'spin_1%s' % matrix_name).setValue(column_number)
        getattr(self, 'spin_2%s' % matrix_name).setValue(row_number)
        row_number = getattr(self, 'spin_1%s' % matrix_name).value()
        column_number = getattr(self, 'spin_2%s' % matrix_name).value()
        for row in range(1, row_number + 1):
            for col in range(1, column_number + 1):
                getattr(self, 'spin_%s%s%s' % (row, col, matrix_name)).show()
                getattr(self, 'spin_%s%s%s' % (row, col, matrix_name)).setValue(matrix[row - 1, col - 1])

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
        if row_number != column_number or getattr(self, 'spin_12%s' % matrix_name).isHidden():
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
            matrix = np.zeros((row_number, column_number))
            for row in range(1, row_number + 1):
                for col in range(1, column_number + 1):
                    matrix[row - 1, col - 1] = getattr(self, 'spin_%s%s%s' % (row, col, matrix_name)).value()
            matrix = np.linalg.matrix_power(matrix, power)
            for row in range(1, row_number + 1):
                for col in range(1, column_number + 1):
                    getattr(self, 'spin_%s%s%s' % (row, col, matrix_name)).setValue(matrix[row - 1, col - 1])

    def determinant(self, matrix_name):
        row_number = getattr(self, 'spin_1%s' % matrix_name).value()
        column_number = getattr(self, 'spin_2%s' % matrix_name).value()
        if row_number != column_number or getattr(self, 'spin_12%s' % matrix_name).isHidden():
            QtWidgets.QMessageBox.critical(self, 'Ошибка', 'Данная операция выполнима только для квадратной матрицы.',
                                           defaultButton=QtWidgets.QMessageBox.Ok)
        else:
            matrix = np.zeros((row_number, column_number))
            for row in range(1, row_number + 1):
                for col in range(1, column_number + 1):
                    matrix[row - 1, col - 1] = getattr(self, 'spin_%s%s%s' % (row, col, matrix_name)).value()
            determinant = round(np.linalg.det(matrix), 3)
            getattr(self, 'label_r%s' % matrix_name).setText('Определитель данной матрицы равен: %s' % determinant)
            getattr(self, 'label_r%s' % matrix_name).show()

    def inverse_matrix(self, matrix_name):
        row_number = getattr(self, 'spin_1%s' % matrix_name).value()
        column_number = getattr(self, 'spin_2%s' % matrix_name).value()
        if row_number != column_number or getattr(self, 'spin_12%s' % matrix_name).isHidden():
            QtWidgets.QMessageBox.critical(self, 'Ошибка', 'Данная операция выполнима только для квадратной матрицы.',
                                           defaultButton=QtWidgets.QMessageBox.Ok)
        else:
            matrix = np.zeros((row_number, column_number))
            for row in range(1, row_number + 1):
                for col in range(1, column_number + 1):
                    matrix[row - 1, col - 1] = getattr(self, 'spin_%s%s%s' % (row, col, matrix_name)).value()
            if np.linalg.det(matrix) == 0:
                QtWidgets.QMessageBox.critical(self, 'Ошибка', 'Определитель матрицы равен 0. Обратной матрицы не существует.',
                                               defaultButton=QtWidgets.QMessageBox.Ok)
            else:
                matrix = np.linalg.inv(matrix)
                for row in range(1, row_number + 1):
                    for col in range(1, column_number + 1):
                        getattr(self, 'spin_%s%s%s' % (row, col, matrix_name)).setValue(matrix[row - 1, col - 1])

    def rank(self, matrix_name):
        row_number = getattr(self, 'spin_1%s' % matrix_name).value()
        column_number = getattr(self, 'spin_2%s' % matrix_name).value()
        matrix = np.zeros((row_number, column_number))
        for row in range(1, row_number + 1):
            for col in range(1, column_number + 1):
                matrix[row - 1, col - 1] = getattr(self, 'spin_%s%s%s' % (row, col, matrix_name)).value()
        getattr(self, 'label_r%s' % matrix_name).setText('Ранг данной матрицы равен %s' % np.linalg.matrix_rank(matrix))
        getattr(self, 'label_r%s' % matrix_name).show()

    def swap(self):     # swap matrices A and B
        row_a = self.spin_1A.value()
        col_a = self.spin_2A.value()
        row_b = self.spin_1B.value()
        col_b = self.spin_2B.value()
        matrix_a = np.zeros((row_a, col_a))  # Создание нулевой матрицы для последующего заполнения
        matrix_b = np.zeros((row_b, col_b))
        for row in range(1, row_a + 1):
            for col in range(1, col_a + 1):
                matrix_a[row - 1, col - 1] = getattr(self, 'spin_%s%sA' % (row, col)).value()
        for row in range(1, row_b + 1):
            for col in range(1, col_b + 1):
                matrix_b[row - 1, col - 1] = getattr(self, 'spin_%s%sB' % (row, col)).value()
        self.spin_1A.setValue(row_b)
        self.spin_2A.setValue(col_b)
        self.spin_1B.setValue(row_a)
        self.spin_2B.setValue(col_a)
        self.hide_cells('A')
        self.hide_cells('B')
        for row in range(1, self.spin_1A.value() + 1):
            for col in range(1, self.spin_2A.value() + 1):
                getattr(self, 'spin_%s%sA' % (row, col)).show()
        for row in range(1, self.spin_1B.value() + 1):
            for col in range(1, self.spin_2B.value() + 1):
                getattr(self, 'spin_%s%sB' % (row, col)).show()
        for row in range(1, self.spin_1A.value() + 1):
            for col in range(1, self.spin_2A.value() + 1):
                getattr(self, 'spin_%s%sA' % (row, col)).setValue(matrix_b[row - 1, col - 1])
        for row in range(1, self.spin_1B.value() + 1):
            for col in range(1, self.spin_2B.value() + 1):
                getattr(self, 'spin_%s%sB' % (row, col)).setValue(matrix_a[row - 1, col - 1])
        self.label_rA.hide()
        self.label_rB.hide()

    def add_dif(self, operation):       # difference and addition of matrices
        row_a = self.spin_1A.value()
        col_a = self.spin_2A.value()
        row_b = self.spin_1A.value()
        col_b = self.spin_2A.value()
        if row_a != row_b or col_a != col_b:
            QtWidgets.QMessageBox.critical(self, 'Ошибка', 'Для выполнения операции создайте две матрицы одинакового размера.',
                                           defaultButton=QtWidgets.QMessageBox.Ok)
        else:
            matrix_a = np.zeros((row_a, col_a))
            matrix_b = np.zeros((row_b, col_b))
            for row in range(1, row_a + 1):
                for col in range(1, col_a + 1):
                    matrix_a[row - 1, col - 1] = getattr(self, 'spin_%s%sA' % (row, col)).value()
            for row in range(1, row_b + 1):
                for col in range(1, col_b + 1):
                    matrix_b[row - 1, col - 1] = getattr(self, 'spin_%s%sB' % (row, col)).value()
            matrix_c = 0
            if operation == 'add':
                matrix_c = matrix_a + matrix_b
            elif operation == 'dif':
                matrix_c = matrix_a - matrix_b
            self.spin_1C.setValue(row_a)
            self.spin_2C.setValue(col_a)
            self.hide_cells('C')
            for row in range(1, row_a + 1):
                for col in range(1, col_a + 1):
                    getattr(self, 'spin_%s%sC' % (row, col)).setValue(matrix_c[row - 1, col - 1])
                    getattr(self, 'spin_%s%sC' % (row, col)).show()

    def matrix_mult(self):
        row_a = self.spin_1A.value()
        col_a = self.spin_2A.value()
        row_b = self.spin_1B.value()
        col_b = self.spin_2B.value()
        if col_a != row_b:
            QtWidgets.QMessageBox.critical(self, 'Ошибка', 'Для перемножения матриц необходимо равенство\nчисла стобцов матрицы А и строк матрицы B.',
                                           defaultButton=QtWidgets.QMessageBox.Ok)
        else:
            matrix_a = np.zeros((row_a, col_a))
            matrix_b = np.zeros((row_b, col_b))
            for row in range(1, row_a + 1):
                for col in range(1, col_a + 1):
                    matrix_a[row - 1, col - 1] = getattr(self, 'spin_%s%sA' % (row, col)).value()
            for row in range(1, row_b + 1):
                for col in range(1, col_b + 1):
                    matrix_b[row - 1, col - 1] = getattr(self, 'spin_%s%sB' % (row, col)).value()
            matrix_c = np.dot(matrix_a, matrix_b)
            self.spin_1C.setValue(row_a)
            self.spin_2C.setValue(col_b)
            self.hide_cells('C')
            for row in range(1, row_a + 1):
                for col in range(1, col_b + 1):
                    getattr(self, 'spin_%s%sC' % (row, col)).setValue(matrix_c[row - 1, col - 1])
                    getattr(self, 'spin_%s%sC' % (row, col)).show()

    def qr_decomposition(self, matrix_name):
        row_number = getattr(self, 'spin_1%s' % matrix_name).value()
        column_number = getattr(self, 'spin_2%s' % matrix_name).value()
        if row_number != column_number or getattr(self, 'spin_12%s' % matrix_name).isHidden():
            QtWidgets.QMessageBox.critical(self, 'Ошибка', 'Данная операция выполнима только для квадратной матрицы.',
                                           defaultButton=QtWidgets.QMessageBox.Ok)
        else:
            matrix_a = np.zeros((row_number,  column_number))
            for row in range(1,  row_number + 1):
                for col in range(1, column_number + 1):
                    matrix_a[row - 1, col - 1] = getattr(self, 'spin_%s%s%s' % (row, col, matrix_name)).value()
            if np.linalg.det(matrix_a) == 0:
                QtWidgets.QMessageBox.critical(self, 'Ошибка', 'Определитель равен 0. QR-разложение невозможно.',
                                               defaultButton=QtWidgets.QMessageBox.Ok)
            else:
                matrix_q, matrix_r = np.linalg.qr(matrix_a)
                self.hide_qr()
                for row in range(1, row_number + 1):
                    for col in range(1, column_number + 1):
                        getattr(self, 'label_%s%sq' % (row, col)).setNum(round(matrix_q[row - 1, col - 1], 3))
                        self.label_q.setText('{} = Q x R, где Q = '.format(matrix_name))
                        self.label_q.show()
                        getattr(self, 'label_%s%sq' % (row, col)).show()
                for row in range(1, row_number + 1):
                    for col in range(1, column_number + 1):
                        getattr(self, 'label_%s%sr' % (row, col)).setNum(round(matrix_r[row - 1, col - 1], 3))
                        self.label_r.setText('R = ')
                        self.label_r.show()
                        getattr(self, 'label_%s%sr' % (row, col)).show()

    def choleskiy(self, matrix_name):
        row_number = getattr(self, 'spin_1%s' % matrix_name).value()
        column_number = getattr(self, 'spin_2%s' % matrix_name).value()
        matrix = np.zeros((row_number, column_number))
        for row in range(1, row_number + 1):
            for col in range(1, column_number + 1):
                matrix[row - 1, col - 1] = getattr(self, 'spin_%s%s%s' % (row, col, matrix_name)).value()
        if row_number == column_number > 0 and np.array_equal(matrix, np.transpose(matrix)) and np.linalg.det(matrix):
            matrix_l = np.linalg.cholesky(matrix)
            matrix_l_T = np.transpose(matrix_l)
            self.hide_qr()
            for row in range(1, row_number + 1):
                for col in range(1, column_number + 1):
                    getattr(self, 'label_%s%sq' % (row, col)).setNum(round(matrix_l[row - 1, col - 1], 3))
                    self.label_q.setText('{} = L x L^T, где L = '.format(matrix_name))
                    self.label_q.show()
                    getattr(self, 'label_%s%sq' % (row, col)).show()
                    getattr(self, 'label_%s%sr' % (row, col)).setNum(round(matrix_l_T[row - 1, col - 1], 3))
                    self.label_r.setText('LT = ')
                    self.label_r.show()
                    getattr(self, 'label_%s%sr' % (row, col)).show()

        else:
            QtWidgets.QMessageBox.critical(self, 'Ошибка', 'Матрица должна быть симметричной и положительно определённой.',
                                           defaultButton=QtWidgets.QMessageBox.Ok)

    def hide_qr(self):      # hide the field for qr decomposition
        qr = ['q', 'r']
        for row in range(1, 6):
            for col in range(1, 6):
                for k in qr:
                    getattr(self, 'label_%s%s%s' % (row, col, k)).hide()
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

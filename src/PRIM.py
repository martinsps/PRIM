import numpy as np
from utils.utils import data_frame_difference
import pandas as pd

class PRIM:
    """
    
    """
    def __init__(self,
                 input_data,
                 col_output,
                 positive_class,
                 alpha,
                 threshold_box,
                 threshold_global):
        # Initialize attributes
        self.input_data = input_data
        # Size of input data
        self.N = len(input_data.index)
        self.col_output = col_output
        self.positive_class = positive_class
        self.alpha = alpha
        self.threshold_box = threshold_box
        self.threshold_global = threshold_global
        self.global_mean = self.calculate_mean(input_data)

    def execute(self):
        boxes = []
        end_prim = False
        current_data = self.input_data
        while not end_prim:
            end_box = False
            box_data = current_data
            box = Box()
            latest_box_mean = 1
            while not end_box:
                # Generate all possible boundaries within current data
                possible_boundaries = self.generate_boundaries(box_data)
                # Choose the best one
                best_boundary = self.select_best_boundary(possible_boundaries, box_data)
                # Eliminate instances of new boundary found
                box_data = self.apply_boundary(best_boundary, box_data)
                box.add_boundary(best_boundary)
                # Check if mean has changed
                mean = self.calculate_mean(box_data)
                # If not, peeling must stop
                if latest_box_mean == mean:
                    end_box = True
                latest_box_mean = mean
                if self.stop_condition_box(box_data) or end_box:
                    end_box = True
                    box, box_data = self.bottom_up_pasting(box, box_data, current_data)
                    self.redundant_input_variables(box, box_data, current_data)
                    box.mean = self.calculate_mean(box_data)
                    boxes.append(box)
                    current_data = self.remove_box(box_data, current_data)
            if self.stop_condition_PRIM(current_data):
                end_prim = True
        for box in boxes:
            for boundary in box.boundary_list:
                print(boundary)
            print(box.mean)

    # def execute_step(self):

    def generate_boundaries(self, data):
        """
        Generates all possible boundaries to choose from
        within the data, according to PRIM's algorithm.
        Three types of columns are considered:
        1. Numeric
        2. Categorical
        3. Ordinal (not yet)
        :return: A list of boundaries
        """
        boundaries = []
        columns = list(data.columns)
        columns.remove(self.col_output)
        for col in columns:
            col_type = data[col].dtype
            # Numeric variables
            if col_type == "int64" or col_type == "float64":
                quantile_bottom = np.quantile(data[col], self.alpha)
                quantile_top = np.quantile(data[col], 1 - self.alpha)
                boundary_bottom = Boundary(col, quantile_bottom, '>=')
                boundary_top = Boundary(col, quantile_top, '<=')
                boundaries.append(boundary_bottom)
                boundaries.append(boundary_top)
            # Categorical
            elif col_type == "object" or col_type == "category":
                column = data[col].astype("category")
                levels = column.unique()
                for level in levels:
                    boundary_equal = Boundary(col, level, '!=')
                    boundaries.append(boundary_equal)
            # TODO: ordinal
        return boundaries

    def select_best_boundary(self, boundaries, data):
        best_boundary = None
        best_mean = -1
        for boundary in boundaries:
            mean = self.get_output_mean(boundary, data)
            if mean > best_mean:
                best_mean = mean
                best_boundary = boundary
        return best_boundary

    def get_output_mean(self, boundary, data):
        data_trimmed = data
        data_trimmed = self.apply_boundary(boundary, data_trimmed)
        if len(data_trimmed) == 0:
            return 0
        return self.calculate_mean(data_trimmed)

    # TODO: apply patience (section 8.2)
    def apply_boundary(self, boundary, data):
        if boundary.operator == ">=":
            data = data[data[boundary.variable_name] >= boundary.value]
        elif boundary.operator == "<=":
            data = data[data[boundary.variable_name] <= boundary.value]
        # elif boundary.operator == "=":
        else:
            data = data[data[boundary.variable_name] != boundary.value]
        return data

    def remove_box(self, box_data, data):
        return data_frame_difference(data, box_data)

    def calculate_mean(self, box_data):
        mean = len(box_data[box_data[self.col_output] == self.positive_class].index) / len(box_data.index)
        return mean

    def calculate_box_mean(self, box, data):
        data_box = data
        data_box = self.apply_box(box, data_box)
        return self.calculate_mean(data_box)

    def apply_box(self, box, data):
        for boundary in box.boundary_list:
            data = self.apply_boundary(boundary, data)
        return data

    def bottom_up_pasting(self, box, box_data, data):
        # Number of observations for pasting with real variables
        end_pasting = False
        data_enlarged = box_data
        while not end_pasting:
            boundary, mean_gain = self.select_best_pasting(box, data_enlarged, data)
            if mean_gain > 0:
                box.add_boundary(boundary)
                data_enlarged = self.apply_pasting(box, data_enlarged, data)
            else:
                end_pasting = True
        return box, data_enlarged

    def select_best_pasting(self, box, box_data, data):
        best_mean_gain = 0
        best_pasting = None
        for boundary in box.boundary_list:
            box_aux = Box.box_copy(box)
            # For categorical, the pasting is just eliminating the condition
            if boundary.operator == "!=":
                box_aux.boundary_list.remove(boundary)
                new_boundary = boundary
            else:
                new_boundary = self.generate_pasting_boundary(boundary, data, box_data)
                if new_boundary.value == Boundary.all:
                    box_aux.boundary_list.remove(boundary)
                else:
                    box_aux.add_boundary(new_boundary)
            mean_gain = self.calculate_mean(self.apply_box(box_aux, data)) - self.calculate_mean(box_data)
            if mean_gain > best_mean_gain:
                best_mean_gain = mean_gain
                best_pasting = new_boundary
        return best_pasting, best_mean_gain

    def generate_pasting_boundary(self, boundary, data, box_data):
        n_box = int(self.alpha * len(box_data.index))
        if boundary.operator == "<=":
            max_value_in_box = max(box_data[boundary.variable_name])
            ordered_values = data[data[boundary.variable_name] > max_value_in_box][boundary.variable_name].\
                sort_values()
            # If number of elements is higher than the ones which will
            # be added, we just eliminate the boundary (return "all")
            if len(ordered_values) <= n_box:
                new_boundary = Boundary(boundary.variable_name, Boundary.all, boundary.operator)
            # If not, we take the value that the n_box(th) element has
            else:
                limit = ordered_values.iloc[n_box]
                new_boundary = Boundary(boundary.variable_name, limit, "<=")
        # boundary.operator == ">="
        else:
            min_value_in_box = min(box_data[boundary.variable_name])
            ordered_values = data[data[boundary.variable_name] < min_value_in_box][boundary.variable_name].\
                sort_values(ascending=False)
            # If number of elements is higher than the ones which will
            # be added, we just eliminate the boundary (return "all")
            if len(ordered_values) <= n_box:
                new_boundary = Boundary(boundary.variable_name, Boundary.all, boundary.operator)
            else:
                limit = ordered_values.iloc[n_box]
                new_boundary = Boundary(boundary.variable_name, limit, ">=")
        return new_boundary

    def apply_box(self, box, data):
        data_trimmed = data
        for boundary in box.boundary_list:
            data_trimmed = self.apply_boundary(boundary, data_trimmed)
        return data_trimmed

    def apply_pasting(self, box, box_data, data):
        data_applied_box = self.apply_box(box, data)
        # We add the "new" elements to box_data
        box_data = pd.concat([box_data, data_frame_difference(data_applied_box,box_data)])
        # print(box_data)
        return box_data

    def redundant_input_variables(self, box, box_data, current_data):
        pass

    def stop_condition_PRIM(self, data):
        """
        Determines if PRIM has ended, that is, every subgroup has been
        found given initial parameters and conditions.
        :return: True if stop condition has been reached, False if it has not
        """
        mean = self.calculate_mean(data)
        support = len(data.index) / self.N
        # print(mean, self.global_mean, support, self.threshold_global)
        return mean < self.global_mean or support < self.threshold_global

    def stop_condition_box(self, box_data):
        """
        Determines if this iteration of peeling has ended, that is,
        a new box has been found given algorithm parameters and conditions.
        :return: True if stop condition has been reached, False if it has not
        """
        # Determine if box_data support is below the threshold_box
        support = len(box_data.index) / self.N
        return support <= self.threshold_box




class Box:
    def __init__(self):
        self.boundary_list = []
        self.mean = 0

    @staticmethod
    def box_copy(box):
        new_box = Box()
        for boundary in box.boundary_list:
            new_box.add_boundary(boundary)
        return new_box

    def add_boundary(self, boundary):
        if not isinstance(boundary, Boundary):
            raise TypeError('Object of type Boundary expected, \
                     however type {} was passed'.format(type(boundary)))
        add = True
        for bound in self.boundary_list:
            # 1. If any boundary with the same real variable already exists, leaves
            # the more restrictive one (the new one)
            if boundary.variable_name == bound.variable_name and boundary.operator != "!=" and \
                    boundary.operator == bound.operator:
                self.boundary_list.remove(bound)
                # 2. If we try to add a boundary with the value of "all", we don't add it (the one existent
                # it is eliminated by first condition)
                if boundary.value == Boundary.all:
                    add = False
            # If we add a categorical boundary with the same variable and same value, we eliminate the existent
            # one and we don't add the new one
            if boundary.value == bound.value and boundary.variable_name == bound.variable_name and \
                    boundary.operator == "!=":
                add = False
                self.boundary_list.remove(bound)
        if add:
            self.boundary_list.append(boundary)


class Boundary:
    """
    Tuple of:
    -variable_name: Name of the field
    -value: Value of the variable where the boundary is built from
    -operator: Could be '>=', '<=' or '!='
    """
    # Defined for boundaries used to represent all values
    all = "All"

    def __init__(self, variable_name, value, operator):
        self.variable_name = variable_name
        self.value = value
        self.operator = operator

    def __str__(self):
        return "Boundary: %s %s %s" % (self.variable_name, self.operator, str(self.value))

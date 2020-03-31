import numpy as np
from utils.utils import data_frame_difference

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
            # print(current_data)
            while not end_box:
                # print(box_data)
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
        return len(data_trimmed[data_trimmed[self.col_output] == self.positive_class].index) / len(data_trimmed.index)

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

    # def apply_boundary_inverse(self, boundary, data):
    #     if boundary.operator == ">=":
    #         data = data[data[boundary.variable_name] < boundary.value]
    #     elif boundary.operator == "<=":
    #         data = data[data[boundary.variable_name] > boundary.value]
    #     # elif boundary.operator == "=":
    #     else:
    #         data = data[data[boundary.variable_name] == boundary.value]
    #     return data

    def remove_box(self, box_data, data):
        # for boundary in box.boundary_list:
        #     data = self.apply_boundary_inverse(boundary, data)
        # return data
        # Difference of data frames
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

    # def bottom_up_pasting(self, box, box_data, data):
    #     # Number of observations for pasting with real variables
    #     n_box = self.alpha * len(box_data.index)
    #     end_pasting = False
    #     data_enlarged = box_data
    #     while not end_pasting:
    #         boundary, mean_gain = self.select_best_pasting(box, box_data, data)
    #         if mean_gain > 0:
    #             box.add_boundary(boundary)
    #             data_enlarged = self.apply_pasting(boundary, data_enlarged)
    #     return box, data_enlarged
    #
    # def select_best_pasting(self, box, box_data, data):
    #     pass
    #
    # def apply_pasting(self, boundary, data):
    #     pass

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

    def add_boundary(self, boundary):
        if not isinstance(boundary, Boundary):
            raise TypeError('Object of type Boundary expected, \
                     however type {} was passed'.format(type(boundary)))
        # If any boundary with the same real variable already exists, leaves
        # the more restrictive one (the new one)
        for bound in self.boundary_list:
            if boundary.variable_name == bound.variable_name and boundary.operator != "!=":
                self.boundary_list.remove(bound)
        self.boundary_list.append(boundary)


class Boundary:
    """
    Tuple of:
    -variable_name: Name of the field
    -value: Value of the variable where the boundary is built from
    -operator: Could be '>=', '<=' or '!='
    """
    def __init__(self, variable_name, value, operator):
        self.variable_name = variable_name
        self.value = value
        self.operator = operator

    def __str__(self):
        return "Boundary: %s %s %s" % (self.variable_name, self.operator, str(self.value))

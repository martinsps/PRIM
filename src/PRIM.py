import numpy as np
from utils import data_frame_difference
import pandas as pd


class PRIM:
    """
    Class that represents the Patient Rule Induction Method (PRIM)
    algorithm, used for subgroup discovery.
    It is initialized with the following parameters:
    1. input_data: Data frame object with the input data in which subgroups
    are going to be discovered
    2. col_output: Name of the column that represents the output class value
    3. positive_class: Name of the class value chosen to be positive (the one
    that the algorithm is going to find subgroups)
    4. alpha: Parameter (> 0 and < 1) that sets the degree of "patience"
    5. threshold_box: Parameter that sets the minimum size of the boxes found
    6. threshold_global: Parameter that sets the minimum size of the data yet to
    be analyzed
    7. min_mean: Minimum mean (presence of positive class) for the subgroups discovered
    8. ordinal_columns: Dictionary with the names of the ordinal columns as "keys"
    and the ordered list of values of those columns as "values"
    """

    def __init__(self,
                 input_data,
                 col_output,
                 positive_class,
                 alpha=0.1,
                 threshold_box=0.1,
                 threshold_global=0.2,
                 min_mean=None,
                 ordinal_columns=None):
        """
        Initializes the algorithm object
        :param input_data: Data frame object with the input data in which subgroups are going to be discovered
        :param col_output: Name of the column that represents the output class value
        :param positive_class: Name of the class value chosen to be positive (the one that the algorithm is
        going to find subgroups)
        :param alpha: Parameter (> 0 and < 1) that sets the degree of "patience"
        :param threshold_box: Parameter that sets the minimum size of the boxes found
        :param threshold_global: Parameter that sets the minimum size of the data yet to be analyzed
        :param min_mean: Minimum mean (presence of positive class) for the subgroups discovered
        :param ordinal_columns: Dictionary with the names of the ordinal columns as "keys"
        and the ordered list of values of those columns as "values"
        """
        # Initialize attributes
        self.current_data = input_data
        # Size of input data
        self.N = len(input_data.index)
        self.col_output = col_output
        self.positive_class = positive_class
        self.alpha = alpha
        self.threshold_box = threshold_box
        self.threshold_global = threshold_global
        # If not min mean specified, global mean is used
        if min_mean:
            self.min_mean = min_mean
        else:
            self.min_mean = self.calculate_mean(input_data)
        self.ordinal_columns = ordinal_columns
        # Boxes (subgroups) found
        self.boxes = []

    def execute(self):
        """
        Executes the PRIM's algorithm step by step. Prints the resulting boxes at the end.
        :return:
        """
        end_prim = False
        while not end_prim:
            end_box = False
            box_data = self.current_data
            box = Box()
            while not end_box:
                box, box_data, end_box = self.do_step_box(box, box_data)
                if end_box:
                    box, box_data = self.bottom_up_pasting(box, box_data)
                    box, box_data, variables_eliminated = self.redundant_input_variables(box, box_data)
                    box.mean = self.calculate_mean(box_data)
                    # Only added if it has a minimum box mean
                    if box.mean >= self.min_mean:
                        self.boxes.append(box)
                        self.current_data = self.remove_box(box_data, self.current_data)
            if self.stop_condition_PRIM(box_data):
                end_prim = True
        for i, box in enumerate(self.boxes):
            print("Box", (i + 1), ":")
            for boundary in box.boundary_list:
                print(boundary)
            print(box.mean)
            print("=============")

    def do_step_box(self, box, box_data):
        """
        Performs a step within the finding of a box in the PRIM's algorithm.
        First, it generates all possible boundaries and the chooses the best one
        according to the generated mean. Finally, it updates the box and box_data.
        :param box: Current box that is being built
        :param box_data: The data (left) in which the box is being built
        :return: The box and box_data updated with a new boundary and a boolean
        that indicates if the end condition for the box has been reached
        """
        # Generate all possible boundaries within current data
        possible_boundaries = self.generate_boundaries(box_data)
        # Choose the best one
        best_boundary = self.select_best_boundary(possible_boundaries, box_data)
        # Eliminate instances of new boundary found
        box_data = self.apply_boundary(best_boundary, box_data)
        box.add_boundary(best_boundary)
        return box, box_data, self.stop_condition_box(box_data)

    def generate_boundaries(self, data):
        """
        Generates all possible boundaries to choose from
        within the data, according to PRIM's algorithm.
        Four types of columns are considered:
        1. Numeric (real, not integer)
        2. Ordinal (specified in the parameter "ordinal_columns" of the algorithm)
        3. Integer
        4. Categorical
        :param data: Current data
        :return: A list of boundaries
        """
        boundaries = []
        columns = list(data.columns)
        columns.remove(self.col_output)
        for col in columns:
            col_type = data[col].dtype
            # Real variables
            if col_type == "float64" or col_type == "float32":
                quantile_bottom = np.quantile(data[col], self.alpha)
                quantile_top = np.quantile(data[col], 1 - self.alpha)
                boundary_bottom = Boundary(col, quantile_bottom, '>=')
                boundary_top = Boundary(col, quantile_top, '<=')
                boundaries.append(boundary_bottom)
                boundaries.append(boundary_top)
            # Ordinal (specified in ordinal_columns)
            elif col in self.ordinal_columns:
                column = data[col]
                column = column.astype("category")
                levels = column.unique()
                min_value, max_value = 0, 0
                for value in self.ordinal_columns[col]:
                    if value in levels:
                        min_value = value
                        break
                for value in reversed(self.ordinal_columns[col]):
                    if value in levels:
                        max_value = value
                        break
                # If the value is the same, it means there is only one
                # level left of that variable -> do not add boundary
                if min_value != max_value:
                    boundary_bottom = Boundary(col, min_value, '!=')
                    boundary_top = Boundary(col, max_value, '!=')
                    boundaries.append(boundary_bottom)
                    boundaries.append(boundary_top)
            # Integers (treated similarly to ordinal)
            elif col_type == "int64" or col_type == "int32":
                min_value = min(data[col])
                max_value = max(data[col])
                # If the value is the same, it means there is only one
                # level left of that variable -> do not add boundary
                if min_value != max_value:
                    boundary_bottom = Boundary(col, min_value, '>')
                    boundary_top = Boundary(col, max_value, '<')
                    boundaries.append(boundary_bottom)
                    boundaries.append(boundary_top)

            # Categorical
            elif col_type == "object" or col_type == "category":
                column = data[col].astype("category")
                levels = column.unique()
                for level in levels:
                    boundary_equal = Boundary(col, level, '!=')
                    boundaries.append(boundary_equal)
        return boundaries

    def select_best_boundary(self, boundaries, data):
        """
        Method that chooses the best boundary out of a list of boundaries provided,
        according to the output mean obtained by applying the boundary in the
        current data. If the list is empty, it returns "None".
        :param boundaries: List of boundaries to choose from
        :param data: Current data
        :return: The best boundary found in the list
        """
        best_boundary = None
        best_mean = -1
        for boundary in boundaries:
            mean = self.get_output_mean(boundary, data)
            if mean > best_mean:
                best_mean = mean
                best_boundary = boundary
        return best_boundary

    def get_output_mean(self, boundary, data):
        """
        Method that calculates the output mean (percentage of presence
        of the positive class) in the data after applying a boundary.
        :param boundary: Boundary to be tested
        :param data: Data to be tested
        :return: Mean calculated after applying boundary to data
        """
        data_trimmed = data
        data_trimmed = self.apply_boundary(boundary, data_trimmed)
        if len(data_trimmed) == 0:
            return 0
        return self.calculate_mean(data_trimmed)

    def apply_boundary(self, boundary, data):
        """
        Applies boundary passed to the data, keeping in the data only the examples
        that satisfy the condition in the boundary
        :param boundary:
        :param data:
        :return:
        """
        if boundary.operator == ">=":
            data = data[data[boundary.variable_name] >= boundary.value]
        elif boundary.operator == "<=":
            data = data[data[boundary.variable_name] <= boundary.value]
        elif boundary.operator == ">":
            data = data[data[boundary.variable_name] > boundary.value]
        elif boundary.operator == "<":
            data = data[data[boundary.variable_name] < boundary.value]
        else:
            data = data[data[boundary.variable_name] != boundary.value]
        return data

    def remove_box(self, box_data, data):
        """
        Removes a box from the data, returning the difference of the data with
        the box_data.
        :param box_data:
        :param data:
        :return:
        """
        return data_frame_difference(data, box_data)

    def calculate_mean(self, box_data):
        """
        Calculates the mean (presence of positive class) in the data.
        :param box_data:
        :return:
        """
        mean = len(box_data[box_data[self.col_output] == self.positive_class].index) / len(box_data.index)
        return mean

    def calculate_box_mean(self, box, data):
        """
        Calculates the mean (presence of positive class) in the data
        after applying the box passed as a parameter (applying each
        of the boundaries).
        :param box:
        :param data:
        :return:
        """
        data_box = data
        data_box = self.apply_box(box, data_box)
        return self.calculate_mean(data_box)

    def bottom_up_pasting(self, box, box_data):
        """
        Performs the bottom-up pasting phase of PRIM's algorithm.
        It tries to paste some values to the current boundaries of the
        box that will increase output mean. The pasting
        continues until no mean gain is obtained.
        :param box: Current box (before pasting)
        :param box_data: Current box data (before pasting)
        :return: Box and box data after pasting (may or may not be bigger)
        """
        # Number of observations for pasting with real variables
        end_pasting = False
        data_enlarged = box_data
        while not end_pasting:
            boundary, mean_gain = self.select_best_pasting(box, data_enlarged, self.current_data)
            if mean_gain > 0:
                box.add_boundary(boundary)
                data_enlarged = self.apply_pasting(box, data_enlarged, self.current_data)
            else:
                end_pasting = True
        return box, data_enlarged

    def select_best_pasting(self, box, box_data, data):
        """
        Selects best pasting in the bottom-up pasting phase of
        the PRIM's algorithm from the box's list of boundaries.
        If no good pasting is found, then mean gain will be 0 and
        best pasting will be "None".
        :param box: Current box
        :param box_data: Current box data
        :param data: Current data (also outside the box)
        :return: Best pasting found (new boundary) and its mean gain
        """
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
        """
        It generates a pasting boundary from an already existing numeric boundary
        in a box by doing one of these:
        1. Adding some examples (real variables, operators "<=" and ">=")
        2. Moving boundary value one integer up or down (int variables, operators "<" and ">")
        :param boundary: Existing boundary to be modified
        :param data: Current data (also outside the box)
        :param box_data: Current box data
        :return: New boundary generated
        """
        n_box = int(self.alpha * len(box_data.index))
        if boundary.operator == "<=":
            max_value_in_box = max(box_data[boundary.variable_name])
            ordered_values = data[data[boundary.variable_name] > max_value_in_box][boundary.variable_name]. \
                sort_values()
            # If number of elements is higher than the ones which will
            # be added, we just eliminate the boundary (return "all")
            if len(ordered_values) <= n_box:
                new_boundary = Boundary(boundary.variable_name, Boundary.all, boundary.operator)
            # If not, we take the value that the n_box(th) element has
            else:
                limit = ordered_values.iloc[n_box]
                new_boundary = Boundary(boundary.variable_name, limit, "<=")
        elif boundary.operator == ">=":
            min_value_in_box = min(box_data[boundary.variable_name])
            ordered_values = data[data[boundary.variable_name] < min_value_in_box][boundary.variable_name]. \
                sort_values(ascending=False)
            # If number of elements is higher than the ones which will
            # be added, we just eliminate the boundary (return "all")
            if len(ordered_values) <= n_box:
                new_boundary = Boundary(boundary.variable_name, Boundary.all, boundary.operator)
            else:
                limit = ordered_values.iloc[n_box]
                new_boundary = Boundary(boundary.variable_name, limit, ">=")
        # Integer boundaries, we look for the next value to add
        elif boundary.operator == "<":
            ordered_values = data[data[boundary.variable_name] > boundary.value][boundary.variable_name]. \
                sort_values()
            # It means going up a step would mean add every element
            if len(ordered_values) == 0:
                new_boundary = Boundary(boundary.variable_name, Boundary.all, "<")
            else:
                value = ordered_values.iloc[0]
                new_boundary = Boundary(boundary.variable_name, value, "<")
        elif boundary.operator == ">":
            ordered_values = data[data[boundary.variable_name] < boundary.value][boundary.variable_name]. \
                sort_values(ascending=False)
            # It means going down a step would mean add every element
            if len(ordered_values) == 0:
                new_boundary = Boundary(boundary.variable_name, Boundary.all, ">")
            else:
                value = ordered_values.iloc[0]
                new_boundary = Boundary(boundary.variable_name, value, ">")
        else:
            return None
        return new_boundary

    def apply_box(self, box, data):
        """
        Applies a box to the data, applying every pasting boundary
        of its list.
        :param box: Box to be applied
        :param data: Data trimmed
        :return: Data after applying every boundary of the box
        """
        data_trimmed = data
        for boundary in box.boundary_list:
            data_trimmed = self.apply_boundary(boundary, data_trimmed)
        return data_trimmed

    def apply_pasting(self, box, box_data, data):
        """
        Applies pasting to the box data by adding the examples
        of the whole data that remain after applying the new box.
        :param box: New box after pasting phase
        :param box_data: Box data (before pasting)
        :param data: Current data (also outside the box)
        :return: Box data enlarged by pasting
        """
        data_applied_box = self.apply_box(box, data)
        # We add the "new" elements to box_data
        box_data = pd.concat([box_data, data_frame_difference(data_applied_box, box_data)])
        return box_data

    def redundant_input_variables(self, box, box_data):
        """
        Performs the redundant input variables phase of PRIM's algorithm.
        It calculates the gain of eliminating each variable from the box and
        if one of them is positive, it eliminates the variable from the box and
        executes the pasting of previously eliminated elements. This process is
        repeated until one variable is left in the box or no mean gain is obtained
        from eliminating a new variable.
        :param box: Current box
        :param box_data: Box data
        :return The new box and box_data after the process of redundant input
        variable elimination and a list with the variables eliminated in order
        """
        variables_eliminated = []
        end = False
        while not end:
            mean = self.calculate_mean(box_data)
            best_mean_gain = 0
            variable_best_mean_gain = ""
            box_best_mean_gain = {}
            variable_list = []
            for boundary in box.boundary_list:
                if boundary.variable_name not in variable_list:
                    variable_list.append(boundary.variable_name)
            # If there is only one variable left, it exits
            if len(variable_list) == 1:
                end = True
            else:
                for variable in variable_list:
                    box_aux = Box.box_copy(box)
                    for boundary in box_aux.boundary_list:
                        if boundary.variable_name == variable:
                            box_aux.boundary_list.remove(boundary)
                    mean_gain = self.calculate_box_mean(box_aux, self.current_data) - mean
                    if mean_gain > best_mean_gain:
                        best_mean_gain = mean_gain
                        variable_best_mean_gain = variable
                        box_best_mean_gain = box_aux
                if best_mean_gain > 0:
                    # Eliminate the variable applying pasting with box
                    box_data = self.apply_pasting(box_best_mean_gain, box_data, self.current_data)
                    box = box_best_mean_gain
                    variables_eliminated.append(variable_best_mean_gain)
                else:
                    end = True
        return box, box_data, variables_eliminated

    def stop_condition_PRIM(self, box_data):
        """
        Determines if PRIM has ended, that is, every subgroup has been
        found given initial parameters and conditions.
        The conditions are:
        1. Last box's mean was lower than minimum mean allowed
        2. The support of the remaining data is lower than threshold
        :return: True if stop condition has been reached, False if it has not
        """
        mean = self.calculate_mean(box_data)
        support = len(self.current_data.index) / self.N
        return mean < self.min_mean or support < self.threshold_global

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

    def __str__(self):
        msg = ""
        for ant in self.boundary_list:
            msg += f"{ant} AND "
        return msg[:-4]

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
    -operator: Could be '>=', '<=' (real variables), '!=' (categorical and
    ordinal variables) or '<' and '>' (integer variables)
    """
    # Defined for boundaries used to represent all values
    all = "All"

    def __init__(self, variable_name, value, operator):
        self.variable_name = variable_name
        self.value = value
        self.operator = operator

    def __str__(self):
        return "%s %s %s" % (self.variable_name, self.operator, str(self.value))

import alibi
import numpy
import pandas
import matplotlib
import shap
from XAI.explanation import explanation


class XAI(object):
    """An interface to explaining objects

    This interface can be used to explain black box models using a varienty of explainers.

    Attributes:
        exp (:obj:`explainer`): an object containing all the executed explanations

    """

    def __init__(self, predictor, X, y):
        """A constructor for the XAI class

        Args:
            Predictor (:obj:`callable`): a callback function of the prediciton
            X (:obj:`pandas.dataframe`): features x samples containing training data
            y (:obj:`pandas.dataframe`): features x samples containg testing data

        """
        self._set_predictor(predictor)
        self._set_training_dataset(X)
        self._set_target_dataset(y)
        self.exp = explanation()
        self.__explainer = None

    def _set_predictor(self, callable_function):
        self.__predictor = callable_function

    def _set_training_dataset(self, X):
        self.__X = X

    def _set_target_dataset(self, y):
        self.__y = y

    def _get_explainer(self, **kwargs):
        """A private method fot getting and fitting an explainer.

        This list can be expanded upon, by adding the parameters to **kwargs and tot he if stament below.

        Args:
            **kwargs: all the possible parameters for the explainer and fitter. 

        """
        #All of the kwargs are given default values.
        keys = kwargs.keys()
        exp_type = kwargs["type"] if "type" in keys else "kernel-shap"
        link = kwargs["link"] if "link" in keys else "logit"
        feature_names = kwargs["feature_names"] if "feature_names" in keys else self.__X.columns.to_list()
        target_names = kwargs["target_names"] if "target_names" in keys else None
        categorical_names = kwargs["categorical_names"] if "categorical_names" in keys else None
        task = kwargs["task"] if "task" in keys else "classification"
        seed = kwargs["seed"] if "seed" in keys else None
        distributed_opts = kwargs["distributed_opts"] if "distributed_opts" in keys else None
        model_output = kwargs["model_output"] if "model_output" in keys else "raw"
        check_feature_resolution = kwargs["check_feature_resolution"] if "check_feature_resolution" in keys else True
        low_resolution_threshold = kwargs["low_resolution_threshold"] if "low_resolution_threshold" in keys else 10
        extrapolate_constant = kwargs["extrapolate_constant"] if "extrapolate_constant" in keys else True
        extrapolate_constant_perc = kwargs["extrapolate_constant_perc"] if "extrapolate_constant_perc" in keys else 10.0
        extrapolate_constant_min = kwargs["extrapolate_constant_min"] if "extrapolate_constant_min" in keys else 0.1

        summarise_background = kwargs["summarise_background"] if "summarise_background" in keys else False
        n_background_samples = kwargs["n_background_samples"] if "n_background_samples" in keys else 500

        #Selector for the explainer/fitter.
        if exp_type == "kernel-shap":
            groups = kwargs["groups"] if "groups" in keys else None
            group_names = kwargs["group_names"] if "group_names" in keys else None
            weights = kwargs["weights"] if "weights" in keys else None

            explainer = alibi.explainers.shap_wrappers.KernelShap(self.__predictor,link,feature_names,categorical_names,task,seed,distributed_opts)
            explainer.fit(self.__X,summarise_background,n_background_samples,groups,group_names,weights)
            return explainer

        if exp_type == "tree-shap":
            background_data = self.__X if kwargs.get("use_background_data") else None

            explainer = alibi.explainers.shap_wrappers.TreeShap(self.__predictor,model_output,feature_names,categorical_names,task)
            explainer.fit(background_data, summarise_background, n_background_samples)
            return explainer

        if exp_type == "ale-plot":
            return alibi.explainers.ALE(self.__predictor,feature_names,target_names,check_feature_resolution,low_resolution_threshold,extrapolate_constant, extrapolate_constant_perc, extrapolate_constant_min)

    def reset_explainer(self):
        """Resets the explainer"""
        self.__explainer = None

    def plot_local(self, instance, class_id=0, **kwargs):
        """Plot for a local explaination, currently uses Knernel-SHAP.

        This function can explain an instance using:
        - Force plot
        - Waterfall plot

        Args:
            instance (:obj:`pandas.series`): An instance to explain, length = features.
            class_id (:obj:`int`, optional): Used to explain how it got to that particular class, for regression can be left at 0
            **kwargs: Used to train/fit the explainer list of arguments can be found here: https://docs.seldon.io/projects/alibi/en/latest/api/alibi.explainers.shap_wrappers.html#alibi.explainers.shap_wrappers.KernelShap
                plot-type can be used to select "force" or "waterfall"

        """
        if not self.__explainer:
            self.__explainer = self._get_explainer(**kwargs)

        explanation = self.__explainer.explain(numpy.reshape(instance.to_numpy(),(1,-1)))
        self.exp.add_explanation(explanation)

        instance = instance.to_numpy()
             
        plot_type = kwargs["plot_type"] if "plot_type" in kwargs.keys() else "force"

        if plot_type == "force":
            shap.plots.force(
                explanation.expected_value[class_id],
                explanation.shap_values[class_id][0],
                instance,
                self.__explainer.feature_names,
                matplotlib=True,
                show=False,
                link=self.__explainer.link
            )  
        
        if plot_type == "waterfall":
            shap_exp = shap.Explanation(explanation.shap_values[class_id][0], explanation.expected_value[class_id], feature_names=self.__explainer.feature_names)

            shap.plots.waterfall(
                shap_exp,
                show=False
                )

    def plot_global(self, class_id=None, custom_data=None, **kwargs):
        """Plot for a global explaination, currently uses Knernel-SHAP.

        This function can explain an instance using:
        - summary-bar plot
        - summary-class plot

        The summary-bar plot stacks all the classes on top of eachother. whereas the summary-class does it per class.

        Args:
            class_id (:obj:`int`, optional): The class to summarize, leave None for stacked bar-summary.
            class_id (:obj:`pandas.dataframe`, optional): can be used to summarize custom geiven data like the testing set. shape: (samples x features)
            **kwargs: Used to train/fit the explainer list of arguments can be found here: https://docs.seldon.io/projects/alibi/en/latest/api/alibi.explainers.shap_wrappers.html#alibi.explainers.shap_wrappers.KernelShap

        """
        if not self.__explainer:
            self.__explainer = self._get_explainer(**kwargs)

        data = self.__X
        if type(custom_data) != type(None):
            data = custom_data

        explanation = self.__explainer.explain(data)
        self.exp.add_explanation(explanation)
        
        shap_values = explanation.shap_values
        if class_id != None:
            shap_values = explanation.shap_values[class_id]

        shap.summary_plot(
            shap_values,
            data,
            self.__explainer.feature_names,
            show=False
        )

    def plot_curve(self, **kwargs):
        """Plot for a global explaination, currently uses ALE.

        Creates an Accumulated Local Effects (ALE) plot. More info can be found here: https://docs.seldon.io/projects/alibi/en/latest/methods/ALE.html#Accumulated-Local-Effects

        Args:
            **kwargs: Used to train/fit the explainer list of arguments can be found here: https://docs.seldon.io/projects/alibi/en/latest/api/alibi.explainers.ale.html#alibi.explainers.ale.ALE
                and https://docs.seldon.io/projects/alibi/en/latest/api/alibi.explainers.ale.html#alibi.explainers.ale.ALE

        """
        if not self.__explainer:
            kwargs["type"] = "ale-plot"
            self.__explainer = self._get_explainer(**kwargs)

        keys = kwargs.keys()
        features = kwargs["features"] if "features" in keys else None
        min_bin_points = kwargs["min_bin_points"] if "min_bin_points" in keys else 4
        explanation = self.__explainer.explain(self.__X.to_numpy(), features, min_bin_points)
        self.exp.add_explanation(explanation)

        plot_features = kwargs["plot_features"] if "plot_features" in keys else "all"
        plot_targets = kwargs["plot_targets"] if "plot_targets" in keys else "all"
        plot_n_cols = kwargs["plot_n_cols"] if "plot_n_cols" in keys else 3
        plot_sharey = kwargs["plot_sharey"] if "plot_sharey" in keys else "all"
        plot_constant = kwargs["plot_constant"] if "plot_constant" in keys else False
        plot_ax = kwargs["plot_ax"] if "plot_ax" in keys else None
        plot_line_kw = kwargs["plot_line_kw"] if "plot_line_kw" in keys else None
        plot_fig_kw = kwargs["plot_fig_kw"] if "plot_fig_kw" in keys else None
        return alibi.explainers.ale.plot_ale(explanation, plot_features, plot_targets, plot_n_cols, plot_sharey, plot_constant, plot_ax, plot_line_kw, plot_fig_kw)
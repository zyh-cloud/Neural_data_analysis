"""Phy 2.1.0rc1 plugin: add a one-based, numerically sortable channel column.

Install this file in ``~/.phy/plugins/`` and enable ``FixChSort`` in
``~/.phy/phy_config.py``.
"""

from phy import IPlugin


class FixChSort(IPlugin):
    """Add ``ch_orig`` immediately after Phy's built-in ``ch`` column."""

    def attach_to_controller(self, controller):
        metrics = controller.cluster_metrics

        if "ch" not in metrics:
            raise RuntimeError(
                "FixChSort requires Phy's built-in 'ch' cluster metric."
            )

        # Keep the exact meaning of the built-in ch column, including any
        # channel_mapping/channel_map.npy mapping, convert it to a Python int,
        # and add 1 so the displayed channel number is one-based.
        original_ch_metric = metrics["ch"]

        def ch_orig(cluster_id):
            return int(original_ch_metric(cluster_id)) + 1

        # Plugins are attached before Supervisor is constructed in Phy 2.1.0rc1.
        # Supervisor derives its column order from this dictionary's insertion
        # order, so rebuild it now and place ch_orig directly after ch.
        ordered_metrics = {}
        for name, metric in metrics.items():
            # Make repeated loading/replacement deterministic.
            if name == "ch_orig":
                continue
            ordered_metrics[name] = metric
            if name == "ch":
                ordered_metrics["ch_orig"] = ch_orig

        controller.cluster_metrics = ordered_metrics

        print(
            "[FixChSort] enabled; cluster metric order:",
            list(controller.cluster_metrics),
        )

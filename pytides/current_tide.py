"""Current tides from ellipse or linear constituents."""

from collections import OrderedDict, Iterable
from itertools import takewhile, count
try:
    from itertools import izip, ifilter
except ImportError:  #Python3
    izip = zip
    ifilter = filter
from datetime import datetime, timedelta
import numpy as np
from .astro import astro
from . import constituent
from .tide import Tide
from .tidal_ellipse import ap2ep, ep2ap

d2r, r2d = np.pi / 180.0, 180.0 / np.pi


class CurrentTide(object):
    """Initialize the current tidal model, with the linear velocity
    amplitudes and phases OR a tidal current model.

    Parameters
    ----------

    """
    def __init__(self,
                 constituents=None,
                 u_amplitudes=None,
                 u_phases=None,
                 v_amplitudes=None,
                 v_phases=None,
                 model=None,
                 radians=False):
        if None not in [
                constituents,
                list(u_amplitudes),
                list(u_phases),
                list(v_amplitudes),
                list(v_phases)
        ]:
            if (len(constituents) == len(u_amplitudes) == len(u_phases) ==
                    len(v_amplitudes) == len(v_phases)):
                model = np.zeros(len(u_phases), dtype=CurrentTide.dtype)
                model['constituent'] = np.array(constituents)
                model['u_amplitude'] = np.array(u_amplitudes)
                model['u_phase'] = np.array(u_phases)
                model['v_amplitude'] = np.array(v_amplitudes)
                model['v_phase'] = np.array(v_phases)
            else:
                raise ValueError(
                    "Constituents, amplitudes and phases should all be arrays of equal length."
                )
        elif model is not None:
            if not model.dtype == CurrentTide.dtype:
                raise ValueError(
                    "Model must be a numpy array with dtype == CurrentTide.dtype"
                )
        else:
            raise ValueError(
                "Must be initialised with constituents, amplitudes and phases; or a model."
            )
        if radians:
            model['u_phase'] = r2d * model['u_phase']
            model['v_phase'] = r2d * model['v_phase']
        self.model = model[:]
        self.normalize()

    # This is the data type for the tidal current model

    dtype = np.dtype([('constituent', object), ('u_amplitude', float),
                      ('u_phase', float), ('v_amplitude', float),
                      ('v_phase', float)])

    @classmethod
    def from_ellipse(cls,
                     constituent_names,
                     semimajor_axes,
                     inclinations,
                     phases,
                     eccentricities=None,
                     semiminor_axes=None,
                     radians=False):
        if (eccentricities is None) and (semiminor_axes is None):
            raise Exception(
                'Either eccentricities or semiminor_axes must be given')

        # Make sure everything is iterable, and calculate eccentricities
        # if only semiminor_axes is given.
        if not np.iterable(constituent_names):
            constituent_names = [constituent_names]
        if not np.iterable(semimajor_axes):
            semimajor_axes = np.asarray([semimajor_axes])
        if not np.iterable(inclinations):
            inclinations = np.asarray([inclinations])
        if not np.iterable(phases):
            phases = np.asarray([phases])
        if eccentricities is not None:
            if not np.iterable(eccentricities):
                eccentricities = np.asarray([eccentricities])
        else:
            eccentricities = (np.asarray(semiminor_axes) /
                              np.asarray(semimajor_axes))

        # epp2app expects degrees

        if radians:
            phases = np.degrees(phases)
            inclinations = np.degrees(inclinations)

        # get the linear tide components

        nc = len(constituent_names)
        if ((len(semimajor_axes) != nc) or (len(inclinations) != nc)
                or (len(eccentricities) != nc)):
            raise Exception('All ellipse inputs must have the same size')

        (u_amplitudes, u_phases, v_amplitudes,
         v_phases) = ep2ap(semimajor_axes, inclinations, phases,
                           eccentricities)

        # Add the comstituents in the NOAA constituent list

        constituents = []
        idx = []
        for c in constituent.noaa:
            name = c.name
            if name in constituent_names:
                constituents.append(c)
                idx.append(constituent_names.index(name))

        u_amplitudes = u_amplitudes[idx]
        u_phases = u_phases[idx]
        v_amplitudes = v_amplitudes[idx]
        v_phases = v_phases[idx]

        return cls(constituents=constituents,
                   u_amplitudes=u_amplitudes,
                   u_phases=u_phases,
                   v_amplitudes=v_amplitudes,
                   v_phases=v_phases,
                   model=None,
                   radians=False)

    def prepare(self, *args, **kwargs):
        return CurrentTide._prepare(self.model['constituent'], *args, **kwargs)

    @staticmethod
    def _prepare(constituents, t0, t=None, radians=True):
        """
        Return constituent speed and equilibrium argument at a given time, and constituent node factors at given times.
        Arguments:
        constituents -- list of constituents to prepare
        t0 -- time at which to evaluate speed and equilibrium argument for each constituent
        t -- list of times at which to evaluate node factors for each constituent (default: t0)
        radians -- whether to return the angular arguments in radians or degrees (default: True)
        """
        #The equilibrium argument is constant and taken at the beginning of the
        #time series (t0).  The speed of the equilibrium argument changes very
        #slowly, so again we take it to be constant over any length of data. The
        #node factors change more rapidly.
        if isinstance(t0, Iterable):
            t0 = t0[0]
        if t is None:
            t = [t0]
        if not isinstance(t, Iterable):
            t = [t]
        a0 = astro(t0)
        a = [astro(t_i) for t_i in t]

        #For convenience give u, V0 (but not speed!) in [0, 360)
        V0 = np.array([c.V(a0) for c in constituents])[:, np.newaxis]
        speed = np.array([c.speed(a0) for c in constituents])[:, np.newaxis]
        u = [
            np.mod(
                np.array([c.u(a_i)
                          for c in constituents])[:, np.newaxis], 360.0)
            for a_i in a
        ]
        f = [
            np.mod(
                np.array([c.f(a_i)
                          for c in constituents])[:, np.newaxis], 360.0)
            for a_i in a
        ]

        if radians:
            speed = d2r * speed
            V0 = d2r * V0
            u = [d2r * each for each in u]
        return speed, u, f, V0

    def at(self, t):
        """
        Return the modelled tidal (u,v) current at given times.
        Arguments:
        t -- array of times at which to evaluate the tidal height
        """
        t0 = t[0]
        hours = self._hours(t0, t)
        partition = 240.0
        t = self._partition(hours, partition)
        times = self._times(t0, [(i + 0.5) * partition for i in range(len(t))])
        speed, u, f, V0 = self.prepare(t0, times, radians=True)
        Uamp = self.model['u_amplitude'][:, np.newaxis]
        pu = d2r * self.model['u_phase'][:, np.newaxis]
        Vamp = self.model['v_amplitude'][:, np.newaxis]
        pv = d2r * self.model['v_phase'][:, np.newaxis]

        u_tidal_series = np.concatenate([
            CurrentTide._tidal_series(t_i, Uamp, pu, speed, u_i, f_i, V0)
            for t_i, u_i, f_i in izip(t, u, f)
        ])

        v_tidal_series = np.concatenate([
            CurrentTide._tidal_series(t_i, Vamp, pv, speed, u_i, f_i, V0)
            for t_i, u_i, f_i in izip(t, u, f)
        ])

        return u_tidal_series, v_tidal_series

    @staticmethod
    def _hours(t0, t):
        """
        Return the hourly offset(s) of a (list of) time from a given time.
        Arguments:
        t0 -- time from which offsets are sought
        t -- times to find hourly offsets from t0.
        """
        if not isinstance(t, Iterable):
            return CurrentTide._hours(t0, [t])[0]
        elif isinstance(t[0], datetime):
            return np.array([(ti - t0).total_seconds() / 3600.0 for ti in t])
        else:
            return t

    @staticmethod
    def _partition(hours, partition=3600.0):
        """
    	Partition a sorted list of numbers (or in this case hours).
    	Arguments:
    	hours -- sorted ndarray of hours.
    	partition -- maximum partition length (default: 3600.0)
    	"""
        partition = float(partition)
        relative = hours - hours[0]
        total_partitions = np.ceil(relative[-1] / partition +
                                   10 * np.finfo(np.float).eps).astype('int')
        return [
            hours[np.floor(np.divide(relative, partition)) == i]
            for i in range(total_partitions)
        ]

    @staticmethod
    def _times(t0, hours):
        """
    	Return a (list of) datetime(s) given an initial time and an (list of) hourly offset(s).
    	Arguments:
    	t0 -- initial time
    	hours -- hourly offsets from t0
    	"""
        if not isinstance(hours, Iterable):
            return CurrentTide._times(t0, [hours])[0]
        elif not isinstance(hours[0], datetime):
            return np.array([t0 + timedelta(hours=h) for h in hours])
        else:
            return np.array(hours)

    @staticmethod
    def _tidal_series(t, amplitude, phase, speed, u, f, V0):
        return np.sum(amplitude * f * np.cos(speed * t + (V0 + u) - phase),
                      axis=0)

    def normalize(self):
        """
    	Adapt self.model so that amplitudes are positive and phases are in [0,360) as per convention.
    	"""
        for i, (_, u_amplitude, u_phase, v_amplitude,
                v_phase) in enumerate(self.model):
            if u_amplitude < 0:
                self.model['u_amplitude'][i] = -u_amplitude
                self.model['u_phase'][i] = u_phase + 180.0
            self.model['u_phase'][i] = np.mod(self.model['u_phase'][i], 360.0)

            if v_amplitude < 0:
                self.model['v_amplitude'][i] = -v_amplitude
                self.model['v_phase'][i] = v_phase + 180.0
            self.model['v_phase'][i] = np.mod(self.model['v_phase'][i], 360.0)

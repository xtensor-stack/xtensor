.. Copyright (c) 2016, Johan Mabille, Sylvain Corlay and Wolf Vollprecht

   Distributed under the terms of the BSD 3-Clause License.

   The full license is in the file LICENSE, distributed with this software.

Releasing xtensor
=================

Releasing a new version
-----------------------

From the master branch of xtensor

- Make sure that you are in sync with the master branch of the upstream remote.
- Update the `changelog <https://github.com/xtensor-stack/xtensor/blob/master/docs/source/changelog.rst>`_.
- In file ``xtensor_config.hpp``, set the macros for ``XTENSOR_VERSION_MAJOR``, ``XTENSOR_VERSION_MINOR`` and ``XTENSOR_VERSION_PATCH`` to the desired values.
- In file ``CMakeLists.txt``, update the version of the dependencies and the corresponding variables, e.g. ``xtl_REQUIRED_VERSION``.
- In file ``environment.yml``, update the version of the dependencies including ``xtensor``.
- In file ``README.md``, update the dependencies table.
- Stage the changes (``git add``), commit the changes (``git commit``) and add a tag of the form ``Major.minor.patch``. It is important to not add any other content to the tag name.
- Push the new commit and tag to the main repository. (``git push``, and ``git push --tags``)

Updating the conda-forge recipe
-------------------------------

xtensor has been packaged for the conda package manager. Once the new tag has been pushed on GitHub, edit the conda-forge recipe for xtensor in the following fashion:

- Update the version number to the new ``Major.minor.patch``.
- Set the build number to ``0``.
- Update the hash of the source tarball.
- Check for the versions of the dependencies.
- Optionally, rerender the conda-forge feedstock.

Updating the stable branch
--------------------------

Once the conda-forge package has been updated, update the ``stable`` branch to
the newly added tag.

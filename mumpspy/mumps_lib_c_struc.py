import ctypes

MIN_SUPPORTED_VERSION = "4.10.0"
MAX_SUPPORTED_VERSION = "5.7.999"

AUX_LENGTH = 16 * 1024

c_pointer = ctypes.POINTER

MumpsInt = ctypes.c_int
PMumpsInt = c_pointer(MumpsInt)
MumpsInt8 = ctypes.c_uint64
PMumpsInt8 = c_pointer(MumpsInt8)
MumpsReal = ctypes.c_float
PMumpsReal = c_pointer(MumpsReal)
MumpsComplex = MumpsReal
PMumpsComplex = c_pointer(MumpsComplex)
MumpsReal8 = ctypes.c_double
PMumpsReal8 = c_pointer(MumpsReal8)
MumpsComplex16 = MumpsReal8
PMumpsComplex16 = c_pointer(MumpsComplex16)


def get_all_fields(precision="double"):
    """Get all declared and updated fields for MUMPS C structure for a given precision.

    Args:
        precision (str, optional): request precision (single or double) for real or complex. Defaults to 'double'.

    Returns:
        mumps_c_fields: basics fields for MUMPS C structure
        mumps_c_updates: incremental updates for MUMPS C structure
    """

    cMumpsInt = MumpsInt
    cPMumpsInt = PMumpsInt
    cMumpsInt8 = MumpsInt8
    cPMumpsInt8 = PMumpsInt8
    if precision == "single":
        cMumpsReal = MumpsReal
        cPMumpsReal = PMumpsReal
        cMumpsComplex = MumpsComplex
        cPMumpsComplex = PMumpsComplex
    elif precision == "double":
        cMumpsReal = MumpsReal8
        cPMumpsReal = PMumpsReal8
        cMumpsComplex = MumpsComplex16
        cPMumpsComplex = PMumpsComplex16
    else:
        raise ValueError(f"Precision {precision} not supported!")
    #

    mumps_c_fields = [  # MUMPS 4.10.0
        ("sym", cMumpsInt),
        ("par", cMumpsInt),
        ("job", cMumpsInt),
        ("comm_fortran", cMumpsInt),
        ("icntl", cMumpsInt * 40),
        ("cntl", cMumpsReal * 15),
        ("n", cMumpsInt),
        #
        ("nz_alloc", cMumpsInt),
        # /* Assembled entry */
        ("nz", cMumpsInt),
        ("irn", cPMumpsInt),
        ("jcn", cPMumpsInt),
        ("a", cPMumpsComplex),
        # /* Distributed entry */
        ("nz_loc", cMumpsInt),
        ("irn_loc", cPMumpsInt),
        ("jcn_loc", cPMumpsInt),
        ("a_loc", cPMumpsComplex),
        # /* Element entry */
        ("nelt", cMumpsInt),
        ("eltptr", cPMumpsInt),
        ("eltvar", cPMumpsInt),
        ("a_elt", cPMumpsComplex),
        # /* Ordering, if given by user */
        ("perm_in", cPMumpsInt),
        # /* Orderings returned to user */
        ("sym_perm", cPMumpsInt),
        ("uns_perm", cPMumpsInt),
        # /* Scaling (input only in this version) */
        ("colsca", cPMumpsReal),
        ("rowsca", cPMumpsReal),
        # /* RHS, solution, ouptput data and statistics */
        ("rhs", cPMumpsComplex),
        ("redrhs", cPMumpsComplex),
        ("rhs_sparse", cPMumpsComplex),
        ("sol_loc", cPMumpsComplex),
        ("irhs_sparse", cPMumpsInt),
        ("irhs_ptr", cPMumpsInt),
        ("isol_loc", cPMumpsInt),
        ("nrhs", cMumpsInt),
        ("lrhs", cMumpsInt),
        ("lredrhs", cMumpsInt),
        ("nz_rhs", cMumpsInt),
        ("lsol_loc", cMumpsInt),
        ("schur_mloc", cMumpsInt),
        ("schur_nloc", cMumpsInt),
        ("schur_lld", cMumpsInt),
        ("mblock", cMumpsInt),
        ("nblock", cMumpsInt),
        ("nprow", cMumpsInt),
        ("npcol", cMumpsInt),
        ("info", cMumpsInt * 40),
        ("infog", cMumpsInt * 40),
        ("rinfo", cMumpsReal * 40),
        ("rinfog", cMumpsReal * 40),
        # /* Null space */
        ("deficiency", cMumpsInt),
        ("pivnul_list", cPMumpsInt),
        ("mapping", cPMumpsInt),
        # /* Schur */
        ("size_schur", cMumpsInt),
        ("listvar_schur", cPMumpsInt),
        ("schur", cPMumpsComplex),
        # /* Internal parameters */
        ("instance_number", cMumpsInt),
        ("wk_user", cPMumpsComplex),
        # /* Version number:
        #  length in FORTRAN + 1 for final \0 + 1 for alignment */
        ("version_number", ctypes.c_char * 16),
        # /* For out-of-core */
        ("ooc_tmpdir", ctypes.c_char * 256),
        ("ooc_prefix", ctypes.c_char * 64),
        # /* To save the matrix in matrix market format */
        ("write_problem", ctypes.c_char * 256),
        ("lwk_user", cMumpsInt),
    ]

    mumps_c_updates = {  # incremental updates related to version 4.10.0
        "5.0.0": [
            ("new_after", "icntl", ("keep", cMumpsInt * 500)),
            (
                "new_after",
                "cntl",
                [
                    ("dkeep", cMumpsReal * 130),
                    ("keep8", cMumpsInt8 * 150),
                ],
            ),
            (
                "new_after",
                "rowsca",
                [
                    ("colsca_from_mumps", cMumpsInt),
                    ("rowsca_from_mumps", cMumpsInt),
                ],
            ),
            ("replace", "version_number", ctypes.c_char * 27),
        ],
        "5.1.0": [
            ("replace", "dkeep", cMumpsReal * 230),
            ("new_after", "nz", ("nnz", cMumpsInt8)),
            ("new_after", "nz_loc", ("nnz_loc", cMumpsInt8)),
            ("replace", "version_number", ctypes.c_char * 32),
            (
                "new_after",
                "lwk_user",
                [
                    # /* For save/restore feature */
                    ("save_dir", ctypes.c_char * 256),
                    ("save_prefix", ctypes.c_char * 256),
                ],
            ),
        ],
        "5.2.0": [
            ("replace", "icntl", cMumpsInt * 60),
            ("new_after", "sol_loc", ("rhs_loc", cPMumpsComplex)),
            ("new_after", "isol_loc", ("irhs_loc", cPMumpsInt)),
            (
                "new_after",
                "lsol_loc",
                [
                    ("nloc_rhs", cMumpsInt),
                    ("lrhs_loc", cMumpsInt),
                ],
            ),
            ("replace", "info", cMumpsInt * 80),
            ("replace", "infog", cMumpsInt * 80),
            ("new_after", "save_prefix", ("metis_options", cMumpsInt * 40)),
        ],
        "5.3.0": [
            ("new_after", "n", ("nblk", cMumpsInt)),
            (
                "new_after",
                "a_elt",
                [
                    # /* Matrix by blocks */
                    ("blkptr", cPMumpsInt),
                    ("blkvar", cPMumpsInt),
                ],
            ),
        ],
        "5.7.0": [
            ("new_after", "npcol", ("ld_rhsintr", cMumpsInt)),
            ("new_after", "mapping", ("singular_values", cPMumpsReal)),
            ("delete", "instance_number"),
            ("replace", "ooc_tmpdir", ctypes.c_char * 1024),
            ("replace", "ooc_prefix", ctypes.c_char * 256),
            ("replace", "write_problem", ctypes.c_char * 1024),
            ("replace", "save_dir", ctypes.c_char * 1024),
            ("new_after", "metis_options", ("instance_number", cMumpsInt)),
        ],
    }

    return mumps_c_fields, mumps_c_updates


def version_to_int(v):
    """Convert the version string to an integer ('5.2.1' --> 5002001)."""
    return sum(int(vk) * 10 ** (3 * k) for k, vk in enumerate(v.split(".")[::-1]))


def get_mumps_c_fields(version=None, precision="double"):
    """Return the MUMPS C fields for a given MUMPS version."""

    def update_fields(f, update_f):
        for uf in update_f:
            fk = [k for k, _ in f]
            idx = fk.index(uf[1])

            if uf[0] == "replace":
                f[idx] = (uf[1], uf[2])
            elif uf[0] == "delete":
                del f[idx]
            elif uf[0] == "new_after":
                if isinstance(uf[2], list):
                    f[(idx + 1) : (idx + 1)] = uf[2]
                else:
                    f.insert(idx + 1, uf[2])

        return f

    mumps_c_fields, mumps_c_updates = get_all_fields(precision=precision)

    if version is None:
        fields = mumps_c_fields[:5] + [("aux", ctypes.c_uint8 * AUX_LENGTH)]
    else:
        update_keys = list(mumps_c_updates.keys())
        update_keys.sort()

        vnum = version_to_int(version)
        if vnum < version_to_int(MIN_SUPPORTED_VERSION):
            msg = (
                f"MUMPS version {version} not supported! "
                f"({version} < {MIN_SUPPORTED_VERSION})"
            )
            raise ValueError(msg)

        if vnum > version_to_int(MAX_SUPPORTED_VERSION):
            msg = (
                f"MUMPS version {version} not supported! "
                f"({version} > {MAX_SUPPORTED_VERSION})"
            )
            raise ValueError(msg)

        fields = mumps_c_fields.copy()
        for ukey in update_keys:
            if version_to_int(ukey) > vnum:
                break

            fields = update_fields(fields, mumps_c_updates[ukey])

    return fields


def define_mumps_c_struc(version=None, precision="double"):
    """Return MUMPS_C_STRUC class with given fields."""

    class Mumps_c_struc(ctypes.Structure):
        _fields_ = get_mumps_c_fields(version, precision=precision)

    return Mumps_c_struc

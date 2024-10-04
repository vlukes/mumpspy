import ctypes

MIN_SUPPORTED_VERSION = '4.10.0'
MAX_SUPPORTED_VERSION = '5.7.999'

AUX_LENGTH = 16 * 1024

c_pointer = ctypes.POINTER

MumpsInt = ctypes.c_int
PMumpsInt = c_pointer(MumpsInt)
MumpsInt8 = ctypes.c_uint64
MumpsReal = ctypes.c_double
PMumpsReal = c_pointer(MumpsReal)
MumpsComplex = ctypes.c_double
PMumpsComplex = c_pointer(MumpsComplex)

mumps_c_fields = [  # MUMPS 4.10.0
    ('sym', MumpsInt),
    ('par', MumpsInt),
    ('job', MumpsInt),
    ('comm_fortran', MumpsInt),
    ('icntl', MumpsInt * 40),
    ('cntl', MumpsReal * 15),
    ('n', MumpsInt),
    #
    ('nz_alloc', MumpsInt),
    # /* Assembled entry */
    ('nz', MumpsInt),
    ('irn', PMumpsInt),
    ('jcn', PMumpsInt),
    ('a', PMumpsComplex),
    # /* Distributed entry */
    ('nz_loc', MumpsInt),
    ('irn_loc', PMumpsInt),
    ('jcn_loc', PMumpsInt),
    ('a_loc', PMumpsComplex),
    # /* Element entry */
    ('nelt', MumpsInt),
    ('eltptr', PMumpsInt),
    ('eltvar', PMumpsInt),
    ('a_elt', PMumpsComplex),
    # /* Ordering, if given by user */
    ('perm_in', PMumpsInt),
    # /* Orderings returned to user */
    ('sym_perm', PMumpsInt),
    ('uns_perm', PMumpsInt),
    # /* Scaling (input only in this version) */
    ('colsca', PMumpsReal),
    ('rowsca', PMumpsReal),
    # /* RHS, solution, ouptput data and statistics */
    ('rhs', PMumpsComplex),
    ('redrhs', PMumpsComplex),
    ('rhs_sparse', PMumpsComplex),
    ('sol_loc', PMumpsComplex),
    ('irhs_sparse', PMumpsInt),
    ('irhs_ptr', PMumpsInt),
    ('isol_loc', PMumpsInt),
    ('nrhs', MumpsInt),
    ('lrhs', MumpsInt),
    ('lredrhs', MumpsInt),
    ('nz_rhs', MumpsInt),
    ('lsol_loc', MumpsInt),
    ('schur_mloc', MumpsInt),
    ('schur_nloc', MumpsInt),
    ('schur_lld', MumpsInt),
    ('mblock', MumpsInt),
    ('nblock', MumpsInt),
    ('nprow', MumpsInt),
    ('npcol', MumpsInt),
    ('info', MumpsInt * 40),
    ('infog', MumpsInt * 40),
    ('rinfo', MumpsReal * 40),
    ('rinfog', MumpsReal * 40),
    # /* Null space */
    ('deficiency', MumpsInt),
    ('pivnul_list', PMumpsInt),
    ('mapping', PMumpsInt),
    # /* Schur */
    ('size_schur', MumpsInt),
    ('listvar_schur', PMumpsInt),
    ('schur', PMumpsComplex),
    # /* Internal parameters */
    ('instance_number', MumpsInt),
    ('wk_user', PMumpsComplex),
    # /* Version number:
    #  length in FORTRAN + 1 for final \0 + 1 for alignment */
    ('version_number', ctypes.c_char * 16),
    # /* For out-of-core */
    ('ooc_tmpdir', ctypes.c_char * 256),
    ('ooc_prefix', ctypes.c_char * 64),
    # /* To save the matrix in matrix market format */
    ('write_problem', ctypes.c_char * 256),
    ('lwk_user', MumpsInt),
]


mumps_c_updates = {  # incremental updates related to version 4.10.0
    '5.0.0': [
        ('new_after', 'icntl', ('keep', MumpsInt * 500)),
        ('new_after', 'cntl', [
            ('dkeep', MumpsReal * 130),
            ('keep8', MumpsInt8 * 150),
        ]),
        ('new_after', 'rowsca', [
            ('colsca_from_mumps', MumpsInt),
            ('rowsca_from_mumps', MumpsInt),
        ]),
        ('replace', 'version_number', ctypes.c_char * 27),
    ],
    '5.1.0': [
        ('replace', 'dkeep', MumpsReal * 230),
        ('new_after', 'nz', ('nnz', MumpsInt8)),
        ('new_after', 'nz_loc', ('nnz_loc', MumpsInt8)),
        ('replace', 'version_number', ctypes.c_char * 32),
        ('new_after', 'lwk_user', [
            # /* For save/restore feature */
            ('save_dir', ctypes.c_char * 256),
            ('save_prefix', ctypes.c_char * 256),
        ]),
    ],
    '5.2.0': [
        ('replace', 'icntl', MumpsInt * 60),
        ('new_after', 'sol_loc', ('rhs_loc', PMumpsComplex)),
        ('new_after', 'isol_loc', ('irhs_loc', PMumpsInt)),
        ('new_after', 'lsol_loc', [
            ('nloc_rhs', MumpsInt),
            ('lrhs_loc', MumpsInt),
        ]),
        ('replace', 'info', MumpsInt * 80),
        ('replace', 'infog', MumpsInt * 80),
        ('new_after', 'save_prefix', ('metis_options', MumpsInt * 40)),
    ],
    '5.3.0': [
        ('new_after', 'n', ('nblk', MumpsInt)),
        ('new_after', 'a_elt', [
            # /* Matrix by blocks */
            ('blkptr', PMumpsInt),
            ('blkvar', PMumpsInt),
        ]),
    ],
    '5.7.0': [
        ('new_after', 'npcol', ('ld_rhsintr', MumpsInt)),
        ('new_after', 'mapping', ('singular_values', PMumpsReal)),
        ('delete', 'instance_number'),
        ('replace', 'ooc_tmpdir', ctypes.c_char * 1024),
        ('replace', 'ooc_prefix', ctypes.c_char * 256),
        ('replace', 'write_problem', ctypes.c_char * 1024),
        ('replace', 'save_dir', ctypes.c_char * 1024),
        ('new_after', 'metis_options', ('instance_number', MumpsInt)),
    ],
}


def version_to_int(v):
    """Convert the version string to an integer ('5.2.1' --> 5002001)."""
    return sum(int(vk) * 10**(3*k) for k, vk in enumerate(v.split('.')[::-1]))


def get_mumps_c_fields(version=None):
    """Return the MUMPS C fields for a given MUMPS version."""
    def update_fields(f, update_f):
        for uf in update_f:
            fk = [k for k, _ in f]
            idx = fk.index(uf[1])

            if uf[0] == 'replace':
                f[idx] = (uf[1], uf[2])
            elif uf[0] == 'delete':
                del f[idx]
            elif uf[0] == 'new_after':
                if isinstance(uf[2], list):
                    f[(idx + 1):(idx + 1)] = uf[2]
                else:
                    f.insert(idx + 1, uf[2])

        return f

    if version is None:
        fields = mumps_c_fields[:5] + [('aux', ctypes.c_uint8 * AUX_LENGTH)]

    else:
        update_keys = list(mumps_c_updates.keys())
        update_keys.sort()

        vnum = version_to_int(version)
        if vnum < version_to_int(MIN_SUPPORTED_VERSION):
            msg = (f'MUMPS version {version} not supported! '
                f'({version} < {MIN_SUPPORTED_VERSION})')
            raise ValueError(msg)

        if vnum > version_to_int(MAX_SUPPORTED_VERSION):
            msg = (f'MUMPS version {version} not supported! '
                f'({version} > {MAX_SUPPORTED_VERSION})')
            raise ValueError(msg)

        fields = mumps_c_fields.copy()
        for ukey in update_keys:
            if version_to_int(ukey) > vnum:
                break

            fields = update_fields(fields, mumps_c_updates[ukey])

    return fields


def define_mumps_c_struc(version=None):
    """Return MUMPS_C_STRUC class with given fields."""
    class Mumps_c_struc(ctypes.Structure):
        _fields_ = get_mumps_c_fields(version)

    return Mumps_c_struc

import click
import os
import stat


class Method(click.ParamType):
    def __init__(self):
        self.name = "method"

    def convert(self, value, param, ctx):
        if value in ["mse", "nrmse", "psnr", "ssim"]:
            return value
        else:
            self.fail('Must be one of: mse, nrmse, psnr, ssim.', param, ctx)


class PathList(click.ParamType):
    def __init__(self):
        self.name = "path_list"
        self.path_type = 'File'

    def check_path(self, path, param, ctx):
        try:
            st = os.stat(path)
        except OSError:
            self.fail('%s "%s" does not exist.' % (
                self.path_type,
                path
            ), param, ctx)

        if stat.S_ISDIR(st.st_mode):
            self.fail('%s "%s" is a directory.' % (
                self.path_type,
                path
            ), param, ctx)

        if not os.access(path, os.R_OK):
            self.fail('%s "%s" is not readable.' % (
                self.path_type,
                path
            ), param, ctx)

    def convert(self, value, param, ctx):
        out = []

        if len(value.split(',')) > 1:

            if value[0] is not '[' and value[-1] is not ']':
                self.fail('List of paths must be comma separated with no spaces and encased in [].', param, ctx)

            value = value[1:-1]
            paths = value.split(',')

            for path in paths:
                self.check_path(path, param, ctx)
                out.append(path)

        else:
            self.check_path(value, param, ctx)
            out.append(value)

        return out


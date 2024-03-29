#!/usr/bin/env python
# PYTHON_ARGCOMPLETE_OK
import sys
import ubelt as ub


def main(cmdline=True, **kw):
    """
    kw = dict(command='stats')
    cmdline = False
    """
    modnames = [
        'stack_images',
    ]
    module_lut = {}
    for name in modnames:
        mod = ub.import_module_from_name('kwimage.cli.{}'.format(name))
        module_lut[name] = mod

    # Create a list of all submodules with CLI interfaces
    cli_modules = list(module_lut.values())

    from scriptconfig.modal import ModalCLI
    modal = ModalCLI(description=ub.codeblock(
        '''
        The Kitware Image CLI
        '''))

    def get_version(self):
        import kwimage
        return kwimage.__version__
    modal.__class__.version = property(get_version)

    for cli_module in cli_modules:

        cli_config = None
        if hasattr(cli_module, '_CLI'):
            # Old way
            cli_cls = cli_module._CLI
            cli_cls.CLIConfig.__command__ = cli_cls.name
            assert hasattr(cli_cls, 'CLIConfig'), (
                'We are only supporting scriptconfig CLIs')
            # scriptconfig cli pattern
            cli_config = cli_cls.CLIConfig

            if not hasattr(cli_config, 'main'):
                if hasattr(cli_cls, 'main'):
                    main_func = cli_cls.main
                    # Hack the main function into the config
                    cli_config.main = main_func
                else:
                    raise AssertionError(f'No main function for {cli_module}')
        elif hasattr(cli_module, '__config__'):
            # New way
            cli_config = cli_module.__config__
        elif hasattr(cli_module, '__cli__'):
            # New way
            cli_config = cli_module.__cli__
        else:
            raise NotImplementedError

        # Update configs to have aliases / commands attributes
        # cli_modname = cli_module.__name__
        # cli_rel_modname = cli_modname.split('.')[-1]
        cmdname_aliases = ub.oset()
        alias = getattr(cli_module, '__alias__', getattr(cli_config, '__alias__', []))
        if isinstance(alias, str):
            alias = [alias]
        command = getattr(cli_module, '__command__', getattr(cli_config, '__command__', None))
        if command is not None:
            cmdname_aliases.add(command)
        cmdname_aliases.update(alias)
        # cmdname_aliases.update(cmd_alias.get(cli_modname, []) )
        cmdname_aliases.add(cli_config.__command__)
        primary_cmdname = cmdname_aliases[0]
        secondary_cmdnames = cmdname_aliases[1:]
        cli_config.__command__ = primary_cmdname
        cli_config.__alias__ = secondary_cmdnames
        modal.register(cli_config)

    ret = modal.run()
    return ret


if __name__ == '__main__':
    sys.exit(main())

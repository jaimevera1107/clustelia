# Publicar Clustelia en GitHub y PyPI

## 1. Crear y subir el repositorio (PowerShell)

Instalar Git y GitHub CLI (`gh`) si no están instalados. Iniciar sesión:

```powershell
gh auth login
```

Abrir una terminal en la carpeta extraída `clustelia_starter`:

```powershell
git init -b main
git add .
git commit -m "chore: bootstrap Clustelia 0.0.0 with PyPI trusted publishing"
gh repo create jaimevera1107/clustelia --public --source . --remote origin --push
```

Si Git reclama identidad, configurar `git config --global user.name` y `git config --global user.email` con los valores que se prefieran.

Alternativa: crear un repositorio público y vacío desde https://github.com/new (sin README, .gitignore ni licencia) y hacer `git remote add origin https://github.com/jaimevera1107/clustelia.git; git push -u origin main` después del commit.

## 2. Crear el environment en GitHub

En el repositorio: Settings -> Environments -> New environment -> escribir `pypi`. Opcional pero recomendado: activar aprobación requerida para publicar. Para evitar publicar tags arbitrarios, restringir el despliegue a tags que comiencen por `v`.

## 3. Conectar PyPI antes del primer release

Iniciar sesión en https://pypi.org/manage/account/publishing/ y elegir la creación de un Pending Trusted Publisher para GitHub Actions. Completar:

- PyPI project name: `clustelia`
- GitHub owner: `jaimevera1107`
- Repository name: `clustelia`
- Workflow filename: `publish.yml`
- Environment name: `pypi`

Guardar. El pending publisher no reserva el nombre en PyPI. Es necesario publicar para registrar el proyecto.

## 4. Publicar la primera versión

Opción web: repositorio -> Releases -> Draft a new release -> Create new tag `v0.0.0` (target: `main`) -> Publish release.

Opción CLI:

```powershell
gh release create v0.0.0 --target main --title "Clustelia v0.0.0" --notes "First functional preview: cluster-label summary utilities."
```

GitHub Actions ejecutará `.github/workflows/publish.yml`, comprobará que el tag coincide con `pyproject.toml`, ejecutará pruebas, generará wheel y sdist, y los publicará en PyPI.

Verificar: https://github.com/jaimevera1107/clustelia/actions y https://pypi.org/project/clustelia/

## 5. Releases siguientes

Editar `version` en `pyproject.toml` y `__version__` en `src/clustelia/__init__.py` (ambos deben coincidir), actualizar CHANGELOG, ejecutar tests, crear commit, subirlo y publicar un nuevo release con tag `v<version>`. PyPI no permite sobrescribir versiones ya publicadas.

> Aviso: no se ha elegido una licencia. Elegir y añadir el archivo LICENSE según la licencia que se desee otorgar.
